import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import pandas as pd
import datetime as dt

class Interactive(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.root = parent
        
        self.sim_line_coords = []
        self.visual_volatility = 0.15
        
        self.root.bind('<Control_L>', self.ctrl_motion_handler)
        self.bind('<Control-Motion>', self.ctrl_motion_handler)
        self.root.bind('<KeyRelease-Control_L>', self.ctrl_release_handler)
        self.root.bind('<Escape>', self.reset_handler)
        self.root.bind("<MouseWheel>", self.wheel_handler)
        
        self.bind("<Button-1>", self.left_click_handler)
        
        self.bind('<Motion>', self.debug_coords)
        self.coords_text = tk.StringVar()
        self.coords_label = ttk.Label(self, textvariable=self.coords_text)
        self.coords_label.place(x=0,y=0)
    
    def draw_spy(self):
        h = self.winfo_height()
        w = self.winfo_width()
        
        bounds_past_max_height = h*0.2
        bounds_past_min_height = h*0.8
        self.bounds_frwd_max_height = h*0.1
        self.bounds_frwd_min_height = h*0.9
        self.bounds_past_width = w*0.2
        self.bounds_frwd_width = w*0.6
        self.sim_calendar_days = 180
        
        spy_hist_data = trailing_spy
        self.spy_today = spy_hist_data[-1]
        half_range = max(abs(min(spy_hist_data)-self.spy_today),abs(max(spy_hist_data)-spy_hist_data[-1]))
        spy_max = spy_hist_data[-1] + half_range
        spy_min = spy_hist_data[-1] - half_range
        spy_width_pxls = np.linspace(0,self.bounds_past_width,len(spy_hist_data))
        spy_height_pxls = [(bounds_past_max_height-bounds_past_min_height)*(spy_hist_data[i]-spy_min)/(spy_max-spy_min)+bounds_past_min_height for i in range(len(spy_width_pxls))]
        
        
        
        self.sim_dates = [f for f in [start_date+dt.timedelta(days=d) for d in range(self.sim_calendar_days)] if f.weekday()<5]
        self.fwd_usd_map = lambda y: self.spy_today+(0.5*h-y)/(bounds_past_min_height-bounds_past_max_height)*2*half_range
        self.usd_to_pxl = lambda usd: (bounds_past_max_height-bounds_past_min_height)*(usd-spy_min)/(spy_max-spy_min)+bounds_past_min_height
        self.trading_day_map = lambda x: int((len(self.sim_dates)-1)*max(0,min(1,(x-self.bounds_past_width)/(self.bounds_frwd_width))))
        self.day_to_x = lambda d: d*self.bounds_frwd_width/len(self.sim_dates)+self.bounds_past_width
        self.day_len_pxls = self.bounds_frwd_width/len(self.sim_dates)
        self.dollar_len_pxls = 1/(self.fwd_usd_map(h*0.5)-self.fwd_usd_map(h*0.5+1))
        
        self.line_start = (spy_width_pxls[-1], spy_height_pxls[-1])
        for x0,y0,x1,y1 in [(spy_width_pxls[i],spy_height_pxls[i],spy_width_pxls[i+1],spy_height_pxls[i+1]) for i in range(len(spy_width_pxls))[:-1]]:
            self.create_line(x0,y0,x1,y1, tags='spy_historic', fill='white')
            
        # bounding box - probably temporary
        self.create_line(self.bounds_past_width,self.bounds_frwd_max_height,self.bounds_past_width+self.bounds_frwd_width,self.bounds_frwd_max_height,tags='temp',fill='green')
        self.create_line(self.bounds_past_width,self.bounds_frwd_min_height,self.bounds_past_width+self.bounds_frwd_width,self.bounds_frwd_min_height,tags='temp',fill='green')
        #self.create_line(self.bounds_past_width,self.bounds_frwd_min_height,self.bounds_past_width,self.bounds_frwd_max_height,fill='green')
        self.create_line(self.bounds_past_width+self.bounds_frwd_width,self.bounds_frwd_min_height,self.bounds_past_width+self.bounds_frwd_width,self.bounds_frwd_max_height,tags='temp',fill='green')
        return
    
    def debug_coords(self, event):
        self.delete('test')
        
        trading_days =  self.trading_day_map(event.x)
        hover_date = self.sim_dates[trading_days]
        hover_price = self.fwd_usd_map(event.y)
        hover_return = hover_price/self.spy_today-1
        self.coords_text.set(f'X={event.x}\nY={event.y}\nTrading Days={trading_days}\nDate={hover_date}\nSPY={hover_price}\nSPY Return={hover_return:.2%}\nVolatility (annl.)={self.visual_volatility:.2%}')
        
    
    def left_click_handler(self, event):
        print(event.x, event.y)
        #self.create_oval(event.x-5, event.y-5,event.x+5, event.y+5, tags='last_clicked', fill='white')
        
    def wheel_handler(self,event):
        self.visual_volatility = min(0.7,max(0.05,self.visual_volatility+event.delta/12000))
        
    def ctrl_motion_handler(self, event):
        self.delete('pending_sim_line')
        if (event.x > self.line_start[0]) and (event.y>self.bounds_frwd_max_height) and (event.y<self.bounds_frwd_min_height) and (event.x<self.bounds_past_width+self.bounds_frwd_width):
            self.create_line(self.line_start[0],self.line_start[1],event.x,event.y, tags='pending_sim_line', fill='blue')
            trading_days =  int((event.x-self.line_start[0])/self.day_len_pxls)
            spy_at_start = self.fwd_usd_map(self.line_start[1])
            slope = (event.y-self.line_start[1])/max(1,trading_days)
            pxl_spread = [0]+[self.dollar_len_pxls*spy_at_start*self.visual_volatility/(250/(t+1))**0.5 for t in range(trading_days)]
            for i in range(len(pxl_spread)-1):
                self.create_line(self.line_start[0]+(i*self.day_len_pxls),
                                 self.line_start[1]+pxl_spread[i]+(i)*slope,
                                 self.line_start[0]+((i+1)*self.day_len_pxls),
                                 self.line_start[1]+pxl_spread[i+1]+(i+1)*slope, 
                                 tags='pending_sim_line', fill='red')
                self.create_line(self.line_start[0]+(i*self.day_len_pxls),
                                 self.line_start[1]-pxl_spread[i]+(i)*slope,
                                 self.line_start[0]+((i+1)*self.day_len_pxls),
                                 self.line_start[1]-pxl_spread[i+1]+(i+1)*slope, 
                                 tags='pending_sim_line', fill='green')
                self.create_line(self.line_start[0]+(i*self.day_len_pxls),
                                 self.line_start[1]+2*pxl_spread[i]+(i)*slope,
                                 self.line_start[0]+((i+1)*self.day_len_pxls),
                                 self.line_start[1]+2*pxl_spread[i+1]+(i+1)*slope, 
                                 tags='pending_sim_line', fill='red')
                self.create_line(self.line_start[0]+(i*self.day_len_pxls),
                                 self.line_start[1]-2*pxl_spread[i]+(i)*slope,
                                 self.line_start[0]+((i+1)*self.day_len_pxls),
                                 self.line_start[1]-2*pxl_spread[i+1]+(i+1)*slope, 
                                 tags='pending_sim_line', fill='green')
            self.debug_coords(event)
        
    def ctrl_release_handler(self, event):
        self.delete('pending_sim_line')
        if (event.x > self.line_start[0]) and (event.y>self.bounds_frwd_max_height) and (event.y<self.bounds_frwd_min_height) and (event.x<self.bounds_past_width+self.bounds_frwd_width):
            release_point = [self.line_start[0],self.line_start[1],event.x,event.y]
            self.sim_line_coords.append(release_point)
            trading_days =  max(1,int((event.x-self.line_start[0])/self.day_len_pxls))
            spy_start = self.fwd_usd_map(self.line_start[1])
            slope = math.log((self.fwd_usd_map(event.y)/self.fwd_usd_map(self.line_start[1])))/trading_days
            rets = np.cumsum(np.random.normal(slope,self.visual_volatility/(250**0.5),trading_days))
            pxl_rets = [self.line_start[1]]+[self.usd_to_pxl(spy_start*math.exp(r)) for r in rets]
            for i in range(len(pxl_rets)-1):
                self.create_line(self.line_start[0]+(i*self.day_len_pxls),
                                 pxl_rets[i],
                                 self.line_start[0]+((i+1)*self.day_len_pxls),
                                 pxl_rets[i+1],
                                 tags='sim_line', fill='white')
            
            self.line_start = (event.x, pxl_rets[-1])
            
    def reset_handler(self, event):
        self.delete('pending_sim_line')
        self.delete('sim_line')
        self.delete('spy_historic')
        self.delete('temp')
        self.sim_line_coords = []
        self.visual_volatility = 0.15
        self.draw_spy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Hedged Equity Playground')
        self.geometry('1920x1080')
        self.resizable(True, True)
        self.state('zoomed')
        self.build_widgets()
        
    def build_widgets(self):  
        self.interactive_chart = Interactive(self, background='#141415')
        self.interactive_chart.pack(expand=True, fill='both')
        self.interactive_chart.update()
        self.interactive_chart.draw_spy()


if __name__ == "__main__":
    data = pd.read_csv('spydata2.csv')
    trailing_spy = list(data['Close'])
    start_date = dt.datetime.strptime(data.iloc[-1]['Date'],'%m/%d/%Y')
    app = App()
    app.mainloop() 