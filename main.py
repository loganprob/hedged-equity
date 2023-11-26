import json
import requests
import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import datetime as dt
from scipy import optimize
from scipy import interpolate
from scipy.stats import norm
import numpy.random as rand

import matplotlib.pyplot as plt


#################### YIELD CURVE MATH ####################


def get_rate(t, param): 
    # Nelson-Siegel-Svensson : https://en.wikipedia.org/wiki/Fixed-income_attribution#Modeling_the_yield_curve
    # NSS (exponential-polynomial) isn't perfect but works here
    # parameters have real-world interpretations, reliable: haven't seen it blow up (Runge) yet like some more accurate fitting function might
    t = max(t, 0.0027) # min term = one day. I should be catching all instances of t==0 before this is called, but just in case... avoids division by 0
    b0,b1,b2,b3,d0,d1 = tuple(param)
    return b0 + b1 * ((1-math.exp(-t/d0))/(t/d0)) + b2 * ((1-math.exp(-t/d0))/(t/d0) - math.exp(-t/d0)) + b3 * ((1-math.exp(-t/d1))/(t/d1) - math.exp(-t/d1))


def get_fwd_rate(fwd_time, term, spot_params):
    a = (1 + get_rate(fwd_time, spot_params))**(fwd_time)
    b = (1 + get_rate(fwd_time + term, spot_params))**(fwd_time + term)
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


#################### EXECUTION ####################

spy_price, div_yld, spot_params, ivol_model = gather_data()
