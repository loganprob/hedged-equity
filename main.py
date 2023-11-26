import json
import requests
import math
import xml.etree.ElementTree as ET
import numpy as np
#import pandas as pd
import datetime as dt
from scipy import optimize
from scipy import interpolate
from scipy.stats import norm
import numpy.random as rand
#import matplotlib.pyplot as plt



#################### YIELD CURVE MATH ####################


def get_rate(t, param): 
    # Nelson-Siegel-Svensson : https://en.wikipedia.org/wiki/Fixed-income_attribution#Modeling_the_yield_curve
    # NSS (exponential-polynomial) isn't perfect but works here
    # parameters have real-world interpretations, reliable: haven't seen it blow up (Runge) yet like some more accurate fitting function might
    t = max(t, 0.0027) # min term = one day. I should be catching all instances of t==0 before this is called, but just in case... avoids division by 0
    b0,b1,b2,b3,d0,d1 = tuple(param)
    return b0 + b1 * ((1-math.exp(-t/d0))/(t/d0)) + b2 * ((1-math.exp(-t/d0))/(t/d0) - math.exp(-t/d0)) + b3 * ((1-math.exp(-t/d1))/(t/d1) - math.exp(-t/d1))


def get_fwd_rate(fwd_time, term, spot_params, rf_adjust=1):
    a = (1 + get_rate(fwd_time, spot_params)*rf_adjust)**(fwd_time)
    b = (1 + get_rate(fwd_time + term, spot_params)*rf_adjust)**(fwd_time + term)
    return (b/a)**(1/term)-1


def fit_curve(known_points):
    guess = [0.02382,0.03002,0.04108,0.05387,0.74504,13.67487] # initial NSS parameter guess, update periodically
    error_func = lambda params, known: sum([(get_rate(t,params)-r)**2 for t,r in known.items()])
    result = optimize.minimize(error_func, guess, args=(known_points))
    return result['x']


def bootstrap_spot(par_params): # need to take another look at, probably make more clear what it's doing & double check logic
    times = np.arange(0.5,3.5,0.5)
    spots = []
    while len(spots)<len(times):
        c = max(0,get_rate(times[len(spots)], par_params)/2)
        spots.append(((1+c)/(1-sum([c/((1+s)**t) for s,t in zip(spots,times)])))**(1/times[len(spots)])-1)
    return dict(zip(times,spots))



#################### OPTION MATH ####################


def ivol_solver(spy, iscall, time, strike, rf, divyld, price):
    # Newton's Method approach to solving for implied vol using market prices
    # https://en.wikipedia.org/wiki/Newton%27s_method , https://en.wikipedia.org/wiki/Greeks_(finance)#Vega
    # orginally saw this approach in some random code on Schwab's retail website, but I can't locate it now
    flag = 1 if iscall else -1
    # attempt at doing less repetitive math in the loop... newton's method converges so fast that this really doesn't improve anything
    lnsk = math.log(spy/strike)
    seqt = spy * math.exp(-divyld * time)
    kert = strike * math.exp(-rf * time)
    sqrt = time**0.5
    i = 0
    max_iter = 10
    guess = 0.5
    while i < max_iter:
        d1 = (lnsk + time * (rf - divyld + 0.5*(guess**2))) / (guess * sqrt)
        d2 = d1 - (guess * sqrt)
        theo_price = flag * (seqt * norm.cdf(flag * d1) - kert * norm.cdf(flag * d2))
        if abs(theo_price - price) < 0.01:
            return guess
        vega = seqt * norm.pdf(d1) * sqrt
        if vega <= 0: # normally this happens when the price approaching lower limit of price (< $0.02) and can't solve... 
            return None # ... the "calibrate_iv_surface" should be excluding these options, but just in case
        guess = guess - (theo_price - price) / vega
        i += 1
    return guess


def calibrate_iv_surface(spy_chain, spy_price, div_yld, spot_params):
    iv_surface_quotes = {}
    for option_quote in spy_chain:
        symbol = option_quote['option'].replace('SPY','')
        expir = dt.datetime.strptime(symbol[:6],'%y%m%d').date()
        if expir <= dt.date.today(): # discard expired (applicable if running on weekends) or 0dte quotes
            continue
        time = (expir-dt.date.today()).days/365
        if time > 2: # discard options longer than 2 years from expiration, not needed here
            continue
        iscall = symbol[6]=='C'
        strike = int(symbol[7:12])+int(symbol[12:])/1000
        if (iscall and strike < spy_price) or (not(iscall) and strike > spy_price): # discard itm options
            continue
        if option_quote['bid_size'] * option_quote['ask_size'] == 0: # discard stale quotes
            continue
        price = (option_quote['bid'] + option_quote['ask'])/2
        if price < 0.02: # discard if too far out of the money or too near expiration, avoids extreme ivol numbers as you approach the lower limit of price
            continue
        # while an 'iv' data point comes with this data, it seems that it's frequently equal to 0 or otherwise unreliable/ambigious
        # don't want to find another free/easy data source, and I do trust the bid and ask, so deciding to just solve for implied volatility locally
        spot = get_rate(time, spot_params)
        ivol = ivol_solver(spy_price, iscall, time, strike, spot, div_yld, price)
        if ivol is not None:
            iv_surface_quotes[(time, math.log(strike/spy_price))] = ivol
    ivol_model = interpolate.RBFInterpolator(np.array(list(iv_surface_quotes.keys())), np.array(list(iv_surface_quotes.values())))
    return ivol_model


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


def rehedging_strike_solver(goal, start_spy, sim_spy, time_elapsed, r, q):
    # returns the strike of a otm call with 1yr to expiration with option price equal to 'goal'
    # I've gone around in circles a couple times trying to solve this analytically, but I don't think it's possible... may revist eventually
    iv = lambda k: forecast_ivol(time_elapsed, 1, math.log(k), math.log(sim_spy/start_spy), ivol_model([[1, math.log(k)]])[0])
    strike_error = lambda k, zero: (math.exp(-q)*norm.cdf((r-q+(0.5*iv(k)**2)-math.log(k))/iv(k)) - k*math.exp(-r)*norm.cdf((r-q-(0.5*iv(k)**2)-math.log(k))/iv(k))-zero)**2
    return round(optimize.minimize(strike_error, x0=[1.1], args=(goal/sim_spy))['x'][0]*sim_spy,2)



#################### DATA GATHERING ####################


def get_spy_dividend_yield():
    response = requests.get('https://www.multpl.com/s-p-500-dividend-yield')
    if response.status_code != 200:  
        print(f'WARNING: Bad response from {response.url}')
        return None
    try: div_yld = float(response.content.decode().split('<div id="current">')[1].split('%')[0].split('>')[-1].replace('\n',''))/100
    except: return None
    return div_yld


def get_spy_option_chain():
    response = requests.get('https://www.cboe.com/delayed_quotes/spy/quote_table/')
    if response.status_code != 200:  
        print(f'WARNING: Bad response from {response.url}')
        return None
    data = json.loads(response.content.decode().split('CTX.contextOptionsData = ')[1].split()[0].replace(';',''))
    spy_price = data['data']['current_price']
    spy_chain = data['data']['options']
    return spy_price, spy_chain


def get_yield_curve():
    url_data = 'daily_treasury_yield_curve'
    url_year = str(dt.date.today().year)
    url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data={url_data}&field_tdr_date_value={url_year}'
    response = requests.get(url) # this request takes like 10 seconds everytime... maybe need a better source, but this isn't HFT
    if response.status_code != 200:  
        print(f'WARNING: Bad response from {response.url}')
        return None
    raw_xml_str = response.content.decode()
    ns_dict = {'main': raw_xml_str.split('xmlns=')[1].split('"')[1], # still haven't found a better solution for parsing out the xml namespaces
               'd': raw_xml_str.split('xmlns:d=')[1].split('"')[1], 
               'm': raw_xml_str.split('xmlns:m=')[1].split('"')[1]}
    root = ET.fromstring(raw_xml_str)
    times = [0.0833, 0.1666, 0.25, 0.3333, 0.5, 1.0, 2.0, 3.0] # only really concerned about fitting the first few years
    return {t: float(r.text)/100 for t, r in zip(times, root.findall('main:entry', ns_dict)[-1].find('main:content/m:properties', ns_dict)[1:])}


def gather_data():
    # this is all very guerilla because I don't want to pay for anything
    print('Gathering Data...')
    spy_price, spy_chain = get_spy_option_chain()
    print(f'SPY Last Price: ${spy_price:.2f}')
    div_yld = get_spy_dividend_yield()
    print(f'SPY Dividend Yield: {div_yld:.2%}')
    par_rates = get_yield_curve()
    print('Fitting Yield Curve...')
    par_params = fit_curve(par_rates)
    spot_params = fit_curve(bootstrap_spot(par_params))
    print('Calibrating Implied Volatility Surface...')
    ivol_model = calibrate_iv_surface(spy_chain, spy_price, div_yld, spot_params)
    
    return spy_price, div_yld, spot_params, ivol_model



#################### OPTION/FUND CLASSES ####################


class Option:
    def __init__(self, iscall, expir, strike):
        self.iscall = iscall
        self.expir = expir
        self.strike = strike
        
    def __str__(self):
        return f"{self.expir} - {'C' if self.iscall else 'P'} - ${self.strike}" # this is just for quality of life while writing/debugging
    
    def mark(self, mark_date, mark_spy, mark_div, mark_rf, mark_iv):
        flag = 1 if self.iscall else -1
        t = (self.expir - mark_date).days/365
        lnsk = math.log(mark_spy/self.strike)
        seqt = mark_spy * math.exp(-mark_div * t)
        kert = self.strike * math.exp(-mark_rf * t)
        sqrt = t**0.5
        d1 = (lnsk + t * (mark_rf - mark_div + 0.5 * (mark_iv ** 2))) / (mark_iv * sqrt)
        d2 = d1 - (mark_iv * sqrt)
        return round(flag * (seqt * norm.cdf(flag * d1) - kert * norm.cdf(flag * d2)),2)
    
    def payoff(self, spy):
        return max(0, (1 if self.iscall else -1) * (spy - self.strike))
    
    def get_ivol(self, mark_date, starting_spy, mark_spy, ivol_model, vol_adjust=1):
        # handles the looking-up/forecasting of ivol in simulations... Fund functions were getting too cluttered
        time_to_expir = (self.expir - mark_date).days/365
        relative_strike = math.log(self.strike/mark_spy)
        starting_ivol_equivalent_timestrike = ivol_model([[time_to_expir, relative_strike]])[0]
        predicted_ivol = forecast_ivol((mark_date - dt.date.today()).days/365, time_to_expir, relative_strike, math.log(mark_spy/starting_spy), starting_ivol_equivalent_timestrike)
        return predicted_ivol * vol_adjust

class Fund:
    def __init__(self, ticker, options):
        # couple things to keep in mind with this model of funds:
        #    nav will always be in terms of SPY, not actual NAV per share
        #    any cash in the fund is ignored, not neccesary and probably impossible to track 
        #    expenses are ignored, the thought is that they affect all funds the same
        #    only built for funds with a single expiration across all held options
        #        eventually I want to change this, and some functions may reflect that foward thinking, but it's not capable to handle this currently
        #        the solution may be to have a parent class that can hold multiple 'funds', each with a single expiration... haven't thought it fully through yet
        self.ticker = ticker
        self.options = options
        self.starting_nav = 0 
        self.rehedge_date = min([o.expir for o in self.options])
        self.rehedge_buffer = 0

        self.rehedge_completed = False
        self.rehedge_options = {}
        self.rehedge_nav_adjustment = 1
    
    def __str__(self):
        return f'{self.ticker}: {[str(o) for o in self.options]}' # this is just for quality of life while writing/debugging
    
    def reset(self):
        self.rehedge_completed = False
        self.rehedge_options = {}
        self.rehedge_nav_adjustment = 1

    def estimate_starting_nav(self, mark_spy, mark_div, spot_params, ivol_model):
        if self.rehedge_date <= dt.date.today():
            print(f'WARNING: {self.ticker} holds options that have expired or are expiring today... will attempt to rehedge, but consider updating data or waiting for fund to trade')
            self.simulate_rehedge(dt.date.today(), mark_spy, mark_spy, mark_div, spot_params, ivol_model, hard_rehedge=True)
        time_to_expir = (self.rehedge_date-dt.date.today()).days/365
        rf = get_rate(time_to_expir, spot_params)
        # consider checking the OCC's FLEX reports at this point for estimate accuracy... might learn something about their pricing model (it's a tree, but black-box) if there are patterns
        # https://www.theocc.com/market-data/market-data-reports/series-and-trading-data/flex-reports
        # https://www.cboe.com/us/options/market_statistics/flex_trade_reports/
        estimate_nav = sum([(o.mark(dt.date.today(), mark_spy, mark_div, rf, ivol_model([[time_to_expir, math.log(o.strike/mark_spy)]])[0]))*q for o,q in self.options.items()])
        self.starting_nav = round(estimate_nav, 2)
    
    # one big change/improvement over the previous version is that all of the rehedging logic and tracking nav/returns has been moved from the monte-carlo simulation to the Fund class
    # keeps the simulation much less cluttered/confusing
    def simulate_rehedge(self, rehedge_date, starting_spy, rehedge_spy, rehedge_div, spot_params, ivol_model, rf_adjust=1, vol_adjust=1, hard_rehedge=False):
        # this is built specifically for the Buffer and Power Buffer funds, but want to make more general eventually
        new_expir = rehedge_date + dt.timedelta(days=365)
        simulated_fwd_time = (rehedge_date-dt.date.today()).days/365
        rehedge_rf = get_rate(1,spot_params)*rf_adjust if simulated_fwd_time==0 else get_fwd_rate(simulated_fwd_time,1,spot_params, rf_adjust=rf_adjust)

        known_strike_options = [Option(True, new_expir, round(rehedge_spy*0.01,2)), Option(False, new_expir, round(rehedge_spy,2)), Option(False, new_expir, round(rehedge_spy*self.rehedge_buffer,2))]
        known_strike_forecast_ivols = [o.get_ivol(rehedge_date, starting_spy, rehedge_spy, ivol_model, vol_adjust=vol_adjust) for o in known_strike_options]
        short_call_goal_val = sum([q*o.mark(rehedge_date, rehedge_spy, rehedge_div, rehedge_rf, iv) for q,o,iv in zip([1,1,-1], known_strike_options, known_strike_forecast_ivols)]) - rehedge_spy*0.99
        short_call_strike = rehedging_strike_solver(short_call_goal_val, starting_spy, rehedge_spy, simulated_fwd_time, rehedge_rf, rehedge_div)

        options = {o:q for o,q in zip(known_strike_options, [1,1,-1])} | {Option(True, new_expir, short_call_strike):-1}

        if hard_rehedge: # not in a simulation, i.e. running the script on the day that something rehedges
            self.options = options
            self.rehedge_date = new_expir
            return
        self.rehedge_completed = True
        self.rehedge_options = options
        return
    
    def simulate_return(self, mark_date, starting_spy, mark_spy, mark_div, spot_params, ivol_model, rf_adjust=1, vol_adjust=1):
        if mark_date == self.rehedge_date:
            payoff_nav = sum([o.payoff(mark_spy)*q for o,q in self.options.items()])
            self.simulate_rehedge(mark_date, starting_spy, mark_spy, mark_div, spot_params, ivol_model, rf_adjust=rf_adjust, vol_adjust=vol_adjust)
            mark_rf = get_fwd_rate((mark_date-dt.date.today()).days/365,(min([o.expir for o in self.rehedge_options])-mark_date).days/365,spot_params,rf_adjust=rf_adjust)
            new_nav = sum([o.mark(mark_date, mark_spy, mark_div, mark_rf, o.get_ivol(mark_date, starting_spy, mark_spy, ivol_model, vol_adjust=vol_adjust))*q for o,q in self.rehedge_options.items()])
            self.rehedge_nav_adjustment = new_nav/payoff_nav
        mark_options = self.rehedge_options if self.rehedge_completed else self.options
        mark_rf = get_fwd_rate((mark_date-dt.date.today()).days/365,(min([o.expir for o in mark_options])-mark_date).days/365,spot_params,rf_adjust=rf_adjust)
        mark_nav = sum([o.mark(mark_date, mark_spy, mark_div, mark_rf, o.get_ivol(mark_date, starting_spy, mark_spy, ivol_model, vol_adjust=vol_adjust))*q for o,q in mark_options.items()])
        fund_return = mark_nav/(self.starting_nav*self.rehedge_nav_adjustment)-1
        return fund_return



#################### MONTE-CARLO ####################


def montecarlo_single(funds, spy_price, div_yld, spot_params, ivol_model, simulation_period=30, rf_vol=0.0, vol_vol=0.0, debug_mode=False, debug_funds = []): # <-- debug stuff is mostly removed
    # if in debug_mode only run one at a time
    sim_start_date = dt.date.today()
    sim_end_date = sim_start_date + dt.timedelta(days=simulation_period)
    if debug_mode:
        sim_dates = [j for j in [sim_start_date + dt.timedelta(days=i+1) for i in range(simulation_period)] if j.weekday()<5]
        debug_funds = list(funds.keys()) if len(debug_funds)==0 else debug_funds
    else:
        sim_dates = sorted(set([*[f.rehedge_date for f in funds.values() if f.rehedge_date < sim_end_date], sim_end_date]))
        
    sim_spy = spy_price
    for i, sim_date in enumerate(sim_dates):
        sim_elapsed_time = (sim_date - dt.date.today()).days/365
        sim_jump_time = (sim_date - sim_dates[i-1]).days/365 if i > 0 else sim_elapsed_time # time since the last repricing date
        # added some imprecise randomness to the system... get adjustments to the yield curve and ivol surface to apply across all the funds
        rf_adjust = math.exp(rand.normal(0,rf_vol/(1/sim_elapsed_time)**0.5))
        vol_adjust = math.exp(rand.normal(0,vol_vol/(1/sim_elapsed_time)**0.5))
        sim_spy_meanret = (1 + get_rate(sim_elapsed_time, spot_params) - div_yld)**(sim_jump_time) - 1
        sim_spy_vol = ivol_model([[sim_elapsed_time, 0]])[0]
        sim_spy *= math.exp(rand.normal(sim_spy_meanret,sim_spy_vol/(1/sim_jump_time)**0.5))

        mark_tickers = debug_funds if debug_mode else [ticker for ticker, fund in funds.items() if fund.rehedge_date == sim_date]
        for ticker in mark_tickers:
            funds[ticker].simulate_return(sim_date, spy_price, sim_spy, div_yld, spot_params, ivol_model, rf_adjust=rf_adjust, vol_adjust=vol_adjust)
        if i==len(sim_dates)-1:
            spy_return = sim_spy/spy_price-1
            fund_returns = {ticker:fund.simulate_return(sim_date, spy_price, sim_spy, div_yld, spot_params, ivol_model, rf_adjust=rf_adjust, vol_adjust=vol_adjust) for ticker, fund in funds.items()}
            return spy_return, fund_returns


def montecarlo_many(funds, spy_price, div_yld, spot_params, ivol_model, simulation_period=30, num_simulations=100, rf_vol=0.0, vol_vol=0.0):
    spy_returns = []
    fund_returns = {ticker: [] for ticker in funds}
    i = 0
    while i < num_simulations:
        s_ret, f_ret = montecarlo_single(funds, spy_price, div_yld, spot_params, ivol_model, simulation_period=simulation_period, rf_vol=rf_vol, vol_vol=vol_vol)
        spy_returns.append(s_ret)
        for f in f_ret:
            fund_returns[f].append(f_ret[f])
            funds[f].reset()
        i += 1
    return spy_returns, fund_returns



#################### EXECUTION ####################


spy_price, div_yld, spot_params, ivol_model = gather_data()

with open('funds.json', 'r') as f:
    funds_data = json.load(f)
funds = {ticker: Fund(ticker, {Option(o['call'], dt.date(*o['expir']), o['strike']):o['qty'] for o in funds_data[ticker]}) for ticker in funds_data}
for fund in funds.values():
    fund.rehedge_buffer = 0.91 if fund.ticker[0]=='B' else 0.85
    fund.estimate_starting_nav(spy_price, div_yld, spot_params, ivol_model)

spy_returns, fund_returns = montecarlo_many(funds, spy_price, div_yld, spot_params, ivol_model, simulation_period=30, num_simulations=1000, rf_vol=0.05, vol_vol=0.1)
    


