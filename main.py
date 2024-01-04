import datetime as dt
import json
import math
import numpy as np
import numpy.random as rand
import requests
from scipy import optimize
from scipy import interpolate
from scipy.stats import norm
import xml.etree.ElementTree as ET
from typing import Callable


#################### YIELD CURVE MATH ####################

def get_rate(spot_params:tuple, borrowing_term:float, forward_time:float=None) -> float:
    # Nelson-Siegel-Svensson : https://en.wikipedia.org/wiki/Fixed-income_attribution#Modeling_the_yield_curve
    # NSS (exponential-polynomial) isn't perfect but works here
    # parameters have real-world interpretations, reliable: haven't seen it blow up (Runge) yet like some more accurate fitting function might
    # forward-rate model comes from CFA L2 textbook
    b0,b1,b2,b3,d0,d1 = spot_params
    t = max(borrowing_term, 0.0027) # min term = one day, else division by zero
    if forward_time is None:
        return b0+b1*((1-math.exp(-t/d0))/(t/d0))+b2*((1-math.exp(-t/d0))/(t/d0)-math.exp(-t/d0))+b3*((1-math.exp(-t/d1))/(t/d1)-math.exp(-t/d1))
    else:
        longer_rate = (get_rate(spot_params, forward_time + t) + 1)**(forward_time + t)
        shorter_rate = (get_rate(spot_params, forward_time) + 1)**(forward_time)
        return (longer_rate/shorter_rate)**(1/t)-1

def fit_yield_curve(known_timerate_pairs:list[tuple]) -> tuple:
    # optimizes NSS params to fit Treasury data
    guess = (0.02382,0.03002,0.04108,0.05387,0.74504,13.67487) # update periodically
    error_function = lambda params, known_rates: sum([(get_rate(params, t) - r)**2 for t,r in known_rates])
    result = optimize.minimize(error_function, guess, args=(known_timerate_pairs))
    return tuple(result['x'])

def bootstrap_spot_curve(par_curve_params:tuple) -> list[tuple]:
    # rewrote this function with an emphasis on clarity... last version was confusing
    # adding more accuracy (maybe) by bootstrapping at a monthly frequency, may have to double check the treasury's methodology to see if this is compatible with their rates
    # https://ebrary.net/14300/economics/inferring_forward_curve
    bootstrap_spots = {}
    for maturity_month in range(1,37): # using months as my iterator/dict-keys to avoid floating point errors when retrieving previously calculated rates 
        coupon = get_rate(par_curve_params, maturity_month/12)/12 # par rate determines value of the semi-annualy coupons, treating face value = 1
        discounted_coupons = sum([coupon/((1+bootstrap_spots[t]/12)**t) for t in range(1,maturity_month)])
        spot_rate = 12*(((1+coupon)/(1-discounted_coupons))**(1/maturity_month)-1)
        bootstrap_spots[maturity_month] = spot_rate
    return [(t/12,r) for t,r in bootstrap_spots.items()]


#################### OPTION/FUND CLASSES & FUNCTIONS ####################

class Option:
    def __init__(self, iscall:bool, expir:dt.date, strike:float):
        self.iscall = iscall
        self.expir = expir
        self.strike = strike
        
    def mark(self, mark_spy:float, mark_div:float, mark_rf:Callable[[float], float], ivol_surface:Callable[[float, float], float], mark_date=dt.date.today()):
        t = (self.expir - mark_date).days/365
        f = 1 if self.iscall else -1
        if t <= 0:
            return max(0, f * (mark_spy - self.strike))
        s = mark_spy
        k = self.strike
        r = mark_rf(t)
        q = mark_div
        iv = ivol_surface(t,k)
        d1 = (math.log(s/k)+t*(r-q+0.5*iv**2))/(iv*t**0.5)
        d2 = d1 - iv*t**0.5
        return f*(s*math.exp(-q*t)*norm.cdf(f*d1)-k*math.exp(-r*t)*norm.cdf(f*d2))
                 
class Fund:
    def __init__(self, ticker:str, options:dict[Option, int], rehedge_behavior:dict):
        self.ticker = ticker
        self.options = options
        self.rehedge_behavior = rehedge_behavior
        self.next_rehedge = min(o.expir for o in self.options)
        self.starting_nav = 0
        self.has_rehedged = False
        self.new_options = None
        self.rehedge_nav_adjustment = 1

    def mark(self, mark_spy:float, mark_div:float, mark_rf:Callable[[float], float], ivol_surface:Callable[[float, float], float], mark_date=dt.date.today()):
        mark_options = self.new_options if self.has_rehedged else self.options
        marked_price = sum([(o.mark(mark_spy,mark_div,mark_rf,ivol_surface,mark_date)*q) for o,q in mark_options.items()])
        if mark_date == self.next_rehedge and not self.has_rehedged:
            return self.rehedge(mark_spy, mark_div, mark_rf, ivol_surface, rehedge_date=mark_date, hard_rehedge=False, payoff_nav=marked_price)
        return marked_price
        
    def rehedge(self, spy_price:float, div_yld:float, rf_rate:Callable[[float], float], ivol_surface:Callable[[float, float], float],rehedge_date=dt.date.today(),hard_rehedge=False, payoff_nav=None):
        rehedge_info = rehedge_solver(self.rehedge_behavior, spy_price, div_yld, rf_rate, ivol_surface, rehedge_date)
        if hard_rehedge:
            self.options = rehedge_info['new_options']
            self.next_rehedge = min(o.expir for o in self.options)
            return
        else:
            self.has_rehedged = True
            self.new_options = rehedge_info['new_options']
            self.rehedge_nav_adjustment = rehedge_info['new_fund_nav']/payoff_nav
            return rehedge_info['new_fund_nav']
    
    def sim_return(self, mark_spy:float, mark_div:float, mark_rf:Callable[[float], float], ivol_surface:Callable[[float, float], float], mark_date=dt.date.today()):
        ending_nav = self.mark(mark_spy,mark_div,mark_rf,ivol_surface,mark_date)
        return ending_nav/(self.starting_nav*self.rehedge_nav_adjustment)-1

    def reset_rehedge(self):
        self.has_rehedged = False
        self.new_options = None
        self.rehedge_nav_adjustment = 1
        return
                 
def bs_option_value(t,f,s,k,r,q,iv):
    d1 = (math.log(s/k)+t*(r-q+0.5*iv**2))/(iv*t**0.5)
    d2 = d1 - iv*t**0.5
    return f*(s*math.exp(-q*t)*norm.cdf(f*d1)-k*math.exp(-r*t)*norm.cdf(f*d2))

def ivol_solver(t,f,s,k,r,q,goal) -> float:
    # see Option class for explanation of arguments
    # Newton's Method approach to solving for implied vol using market prices
    # https://en.wikipedia.org/wiki/Newton%27s_method , https://en.wikipedia.org/wiki/Greeks_(finance)#Vega
    max_iter = 10
    guess = 0.5
    for i in range(max_iter):
        d1 = (math.log(s/k)+t*(r-q+0.5*guess**2))/(guess*t**0.5)
        d2 = d1 - guess*t**0.5
        theo_value = f*(s*math.exp(-q*t)*norm.cdf(f*d1)-k*math.exp(-r*t)*norm.cdf(f*d2))
        if abs(theo_value - goal) < 0.01:
            return guess
        vega = s*math.exp(-q*t)*norm.pdf(d1)*t**0.5
        if vega <= 0: # normally this happens when the price approaching lower limit of price (< $0.02) and can't solve...
            return # also when call stike prices are near $0
        guess -= (theo_value - goal) / vega
    return guess

def calibrate_ivol_surface(option_chain, **curr_data) -> interpolate.RBFInterpolator:
    surface_quotes = {}
    for quote in option_chain:
        symbol = quote['option'].replace('SPY','')
        expir = dt.datetime.strptime(symbol[:6],'%y%m%d').date()
        if expir <= dt.date.today(): # discard expired (applicable if running on weekends) or 0dte quotes
            continue
        time = (expir-dt.date.today()).days/365
        if time > 2: # discard options longer than 2 years from expiration, not needed here
            continue
        iscall = symbol[6]=='C'
        strike = int(symbol[7:12])+int(symbol[12:])/1000
        if (iscall and strike < curr_data['spy_price']) or (not(iscall) and strike > curr_data['spy_price']): # discard itm options
            continue
        if quote['bid_size'] * quote['ask_size'] == 0: # discard stale quotes
            continue
        price = (quote['bid'] + quote['ask'])/2
        if price < 0.02: # discard if too far out of the money or too near expiration, avoids extreme ivol numbers as you approach the lower limit of price
            continue
        spot = get_rate(curr_data['spot_params'], time)
        ivol = ivol_solver(time, 1 if iscall else -1, curr_data['spy_price'], strike, spot, curr_data['dividend_yield'], price)
        if ivol is not None:
            surface_quotes[(time, math.log(strike/curr_data['spy_price']))] = ivol
    ivol_model = interpolate.RBFInterpolator(np.array(list(surface_quotes.keys())), np.array(list(surface_quotes.values())))
    return ivol_model

def rehedge_solver(rehedge_behavior:dict,
                   spy_price:float, 
                   div_yld:float, # may make this dynamic/callable eventually
                   rf_rate:Callable[[float], float], # will call with a single 'borrowing_time' parameter to get a rate
                   ivol_surface:Callable[[float, float], float], # will call with a 'time to expir' and 'logstrike' relative to spy provided
                   rehedge_date=dt.date.today()) -> dict:
    s = spy_price
    target_nav = rehedge_behavior['nav_relative_spy'] * s
    known_strike_option_values = 0.0
    new_options = {}
    for known_strike_option in rehedge_behavior['known_strike_options']:
        t = known_strike_option['time']
        f = 1 if known_strike_option['iscall'] else -1
        k = known_strike_option['relative_strike'] * s
        r = rf_rate(t)
        q = div_yld
        iv = ivol_surface(t,k)
        new_options[Option(known_strike_option['iscall'], rehedge_date + dt.timedelta(days=365*t), k)] = known_strike_option['quantity']
        known_strike_option_values += bs_option_value(t,f,s,k,r,q,iv) * known_strike_option['quantity']
    goal_option_value = (target_nav - known_strike_option_values) / rehedge_behavior['solve_strike_option']['quantity']
    t = rehedge_behavior['solve_strike_option']['time']
    f = 1 if rehedge_behavior['solve_strike_option']['iscall'] else -1
    r = rf_rate(t)
    q = div_yld
    error = lambda k: (bs_option_value(t,f,s,k,r,q,ivol_surface(t,k)) - goal_option_value)**2
    solved_strike = optimize.minimize(error, x0=[s*1.15])['x'][0]
    new_options[Option(rehedge_behavior['solve_strike_option']['iscall'],
                       rehedge_date + dt.timedelta(days=365*t),
                       solved_strike)] = rehedge_behavior['solve_strike_option']['quantity']

    return {'solved_strike': solved_strike, 
            'cap': (solved_strike/s-1 ) * -rehedge_behavior['solve_strike_option']['quantity'],
            'solved_strike_ivol': ivol_surface(t,solved_strike/s),
            'new_fund_nav': known_strike_option_values + bs_option_value(t,f,s,solved_strike,r,q,ivol_surface(t,solved_strike))*rehedge_behavior['solve_strike_option']['quantity'],
            'new_options':new_options}


#################### DATA GATHERING / SETUP ####################

def get_fund_info() -> tuple[dict]:
    response = requests.get('https://raw.githubusercontent.com/loganprob/hedged-equity/main/funds.json')
    if response.status_code != 200:  
        print(f'ERROR: Bad response from Github: {response.status_code} - {response.reason}\n{response.url}\n{response.content}')
        return
    fund_info = response.json()
    response = requests.get('https://raw.githubusercontent.com/loganprob/hedged-equity/main/rehedge_behavior.json')
    if response.status_code != 200:  
        print(f'ERROR: Bad response from Github: {response.status_code} - {response.reason}\n{response.url}\n{response.content}')
        return
    rehedge_definitions = response.json()
    return fund_info, rehedge_definitions

def get_yield_curve() -> list[tuple]:
    base_url = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml'
    url_with_year = f'{base_url}?data=daily_treasury_yield_curve&field_tdr_date_value={dt.date.today().year}'
    response = requests.get(url_with_year)
    if response.status_code != 200:  
        print(f'ERROR: Bad Response from Treasury: {response.status_code} - {response.reason}\n{response.url}\n{response.content}')
        return
    raw_xml_str = response.content.decode()
    ns_dict = {'main': raw_xml_str.split('xmlns=')[1].split('"')[1], # still haven't found a better solution for parsing out the xml namespaces
               'd': raw_xml_str.split('xmlns:d=')[1].split('"')[1], 
               'm': raw_xml_str.split('xmlns:m=')[1].split('"')[1]}
    root = ET.fromstring(raw_xml_str)
    times = [0.0833, 0.1666, 0.25, 0.3333, 0.5, 1.0, 2.0, 3.0] # only concerned about fitting the first few years
    return [(t, float(r.text)/100) for t, r in zip(times, root.findall('main:entry', ns_dict)[-1].find('main:content/m:properties', ns_dict)[1:])]

def get_spy_dividend_yield() -> float:
    response = requests.get('https://www.multpl.com/s-p-500-dividend-yield')
    if response.status_code != 200: 
        print(f'ERROR: Bad response from Multpl.com: {response.status_code} - {response.reason}\n{response.url}\n{response.content}')
        return
    try: div_yld = float(response.content.decode().split('<div id="current">')[1].split('%')[0].split('>')[-1].replace('\n',''))/100
    except: return None
    return div_yld

def get_spy_option_chain() -> tuple:
    response = requests.get('https://www.cboe.com/delayed_quotes/spy/quote_table/')
    if response.status_code != 200:  
        print(f'ERROR: Bad response from CBOE: {response.status_code} - {response.reason}\n{response.url}\n{response.content}')
        return
    try:
        data = json.loads(response.content.decode().split('CTX.contextOptionsData = ')[1].split()[0].replace(';',''))
        spy_price = data['data']['current_price']
        spy_chain = data['data']['options']
        # spy_chain_timestamp = dt.datetime.strptime(data['data']['last_trade_time'].split('T')[0],'%Y-%m-%d').date()
        return spy_price, spy_chain
    except:
        print(f'ERROR: Unable to parse option chain data from CBOE, check website source ({response.url})')
        return
    
def setup() -> dict:
    current_data = {}
    # fund info
    print('Starting execution\n'+'-'*45+'\nAttempting to get fund info...')
    if not (github_response:=get_fund_info()):
        print('Could not fetch fund info from Github, terminating execution')
        return
    current_data['fund_info'], current_data['rehedge_definitions'] = github_response
    # yield curve
    print('Attempting to get yield curve data...')
    if not (treasury_response:=get_yield_curve()):
        print('Could not fetch yield curve data from Treasury, terminating execution')
        return
    print('Fitting yield curve, bootstrapping spot rates...')
    par_params = fit_yield_curve(treasury_response)
    spot_curve_data = bootstrap_spot_curve(par_params)
    current_data['spot_params'] = fit_yield_curve(spot_curve_data)
    # dividend yield
    print('Attempting to get SPY dividend yield...')
    if not (div_yld:=get_spy_dividend_yield()):
        print('Could not fetch dividend yield data from Multp.com, terminating execution')
        return
    current_data['dividend_yield'] = div_yld
    # exchange-traded option chain
    print('Attempting to get SPY option chain data...')
    if not (cboe_response:=get_spy_option_chain()):
        print('Could not fetch option chain data from CBOE, terminating execution')
        return
    current_data['spy_price'], option_chain = cboe_response
    print('Calibrating implied volatility surface...')
    current_data['ivol_model'] = calibrate_ivol_surface(option_chain, **current_data)
    print('Setup successful, ready for analysis...\n')
    return current_data


#################### ANALYSIS/SIMULATION ####################

def calculate_starting_stats(**curr_data): # useful info to see before starting the simulations
    print(f'\nAnalysis as of {dt.datetime.strftime(dt.datetime.now(),"%m/%d/%Y @ %H:%M")}\n')
    print('Estimated caps if strategies rehedged today:\n'+'-'*45)
    for key, long_description in {'innovator_buffer':'Innovator Buffer ("B"-series, 9% buffer)',
                                  'innovator_power_buffer':'Innovator Power Buffer ("P"-series, 15% buffer)',
                                  'innovator_ultra_buffer':'Innovator Ultra Buffer ("U"-series, 5%-30% buffer)',
                                  'innovator_accelerated_buffer':'Innovator Accelerated Buffer ("XB"-series, 9% buffer/2x upside)',}.items():
        cap = rehedge_solver(curr_data['rehedge_definitions'][key], 
                                    curr_data['spy_price'],
                                    curr_data['dividend_yield'],
                                    lambda t: get_rate(curr_data['spot_params'], t),
                                    lambda t, k: curr_data['ivol_model']([[t,math.log(k/curr_data['spy_price'])]])[0])['cap']
        print(f'{long_description:<65}:    {cap:.2%}')
    return

def build_funds(**curr_data):
    print('\nInitializing funds\n'+'-'*45)
    funds = {}
    for ticker, info in curr_data['fund_info'].items():
        funds[ticker] = Fund(ticker, {Option(o['call'], dt.date(*o['expir']), o['strike']):o['qty'] for o in info['holdings']},curr_data['rehedge_definitions'][info['rehedge']])
        if funds[ticker].next_rehedge <= dt.date.today():
            print(f'Available data for fund {ticker} out-of-date, holding options that expired on {funds[ticker].next_rehedge}')
            print(f'Program will attempt to simulate reheding {ticker} as of TODAY. If this behavior is undesired, please check data/contact Logan')
            funds[ticker].rehedge(curr_data['spy_price'],
                                  curr_data['dividend_yield'],
                                  lambda t: get_rate(curr_data['spot_params'], t),
                                  lambda t, k: curr_data['ivol_model']([[t,math.log(k/curr_data['spy_price'])]])[0],
                                  hard_rehedge=True)
        funds[ticker].starting_nav = funds[ticker].mark(curr_data['spy_price'],
                                                        curr_data['dividend_yield'],
                                                        lambda t: get_rate(curr_data['spot_params'], t),
                                                        lambda t, k: curr_data['ivol_model']([[t,math.log(k/curr_data['spy_price'])]])[0])
    return funds

def forecast_ivol(horizon, time_to_expiration, log_strike, spy_logret, starting_ivol):
    # VERY rough model came from personal research, comfortable proceeding regardless because ...
    # ... the goal is to compare relative performance of the funds under different potential ivol surfaces, not predicting future ivol surfaces with high accuracy 
    # will need to revisit one day, but good enough for now... recent paper that looks interesting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4628457
    # numbers derived from oberservations and many multiple regressions on daily SPY option data from 1/2005-11/2021 bought here: https://historicaloptiondata.com/
    # the majority of variation just comes from change in SPY price... anticlimactic 
    # only applicable/tested at forecast horizons at six months or less!
    h = horizon               # year fraction from forecast start to end
    t = time_to_expiration    # year fraction from forecast end to option expiration
    k = log_strike            # ln(strike/(spy at forecast end))
    ds = spy_logret           # ln((spy at forecast end)/(spy at forecast start))
    v0 = starting_ivol        # option implied volatility at forecast start
    t_ = (-0.0203846*h+0.0000255)*t
    k_ = (0.0179652*h-0.0000261)*k
    ds_ = (1.0569557*h-1.9013248)*ds
    i_ = (-0.241939*h*h+0.287693*h)
    dv = math.exp(t_+k_+ds_+i_)
    return v0*dv

class MonteCarlo_Analyzer(dict):
    def append(self, updater:dict[float]) -> None: 
        # I can't find a data structure with this behavior anywhere... I give up... it's a class now
        for key, new_val in updater.items():
            self[key] = self.get(key,[]) + [new_val]
        return
    def get_scores(self, zmin=-1, zmax=1):
        spy_mean = np.mean(self['SPY'])
        spy_std = np.std(self['SPY'])
        zrange_index = [i for i,s in enumerate(self['SPY']) if s>=(spy_mean + spy_std * zmin) and s<=(spy_mean + spy_std * zmax)]
        zrange_returns = {ticker:[r for i,r in enumerate(outcomes) if i in zrange_index] for ticker,outcomes in self.items()}
        return {ticker:np.mean([r-zrange_returns['SPY'][i] for i,r in enumerate(returns)]) for ticker,returns in zrange_returns.items() if ticker!='SPY'}

def montecarlo_single(curr_data:dict, sim_dates:list[dt.date], rf_vol=0.05, vol_vol=0.1):
    sim_spy = curr_data['spy_price']
    for i, sim_date in enumerate(sim_dates):
        sim_elapsed_time = (sim_date - dt.date.today()).days/365
        sim_jump_time = (sim_date - sim_dates[i-1]).days/365 if i > 0 else sim_elapsed_time 
        rf_noise = math.exp(rand.normal(0,rf_vol/(1/sim_elapsed_time)**0.5))
        vol_noise = math.exp(rand.normal(0,vol_vol/(1/sim_elapsed_time)**0.5))
        sim_spy_meanret = (1 + get_rate(curr_data['spot_params'], sim_elapsed_time) - curr_data['dividend_yield'])**(sim_jump_time) - 1
        sim_spy_vol = curr_data['ivol_model']([[sim_elapsed_time, 0]])[0]
        sim_spy *= math.exp(rand.normal(sim_spy_meanret,sim_spy_vol/(1/sim_jump_time)**0.5))
        rf_func = lambda t: get_rate(curr_data['spot_params'], t, sim_elapsed_time) * rf_noise
        vol_func = lambda t,k: forecast_ivol(sim_elapsed_time,t,
                                             math.log(k/sim_spy),
                                             math.log(sim_spy/curr_data['spy_price']), 
                                             curr_data['ivol_model']([[t,math.log(k/sim_spy)]])[0]) * vol_noise
        mark_funds = [fund for fund in curr_data['funds'].values() if fund.next_rehedge==sim_date]
        for fund in mark_funds:
            fund.mark(sim_spy, curr_data['dividend_yield'], rf_func, vol_func, mark_date=sim_date)
        if i == len(sim_dates)-1:
            fund_returns = {}
            for fund in curr_data['funds'].values():
                fund_returns[fund.ticker] = fund.sim_return(sim_spy, curr_data['dividend_yield'], rf_func, vol_func, mark_date=sim_date)
                fund.reset_rehedge()
    return {'SPY':sim_spy/curr_data['spy_price']-1} | fund_returns

def montecarlo_multiple(curr_data:dict, simulation_period:int=30, num_simulations:int=1000):
    print(f'\nStarting Monte Carlo simulation: {num_simulations} runs at ~{simulation_period} calendar days')
    end_date = dt.date.today() + dt.timedelta(days=simulation_period)
    end_date -= dt.timedelta(max(end_date.weekday() - 4, 0)) # round away from weekend
    sim_dates = sorted([*set([fund.next_rehedge for fund in curr_data['funds'].values() if fund.next_rehedge <= end_date]), end_date])
    results = MonteCarlo_Analyzer()
    for i in range(num_simulations):
        results.append(montecarlo_single(curr_data, sim_dates))
    return results

#################### EXECUTION ####################

current_data = setup()
calculate_starting_stats(**current_data)
current_data['funds'] = build_funds(**current_data)
sim_results = montecarlo_multiple(current_data)
print(sim_results.get_scores())
