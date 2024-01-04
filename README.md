_Framework for comparing underlying option exposures of exchange-traded "Buffer" equity funds; running forward-looking simulations and pricing the option holdings under those hypothetical conditions_


1/4/24

Still a mess, but I need to call this a stopping point for now. I've scrapped the whole thing and restarted a number of times. Lots of additional functionality, error handling, and interactivity is sitting in various stages of incompletion and not included here. Many more projects I want to be working on, and I feel like progress has halted and I'm chasing my tail on this one. Perfection is the enemy of good enough. 

General structure is pretty much the same as the last version. Added support for more hedged equity strategies with different rehedging behaviors. Added typing sporadically just to keep things straight in my head (no actual functionality). Passing in risk-free rate parameter and implied volatility parameter as lambda functions to higher order functions is a lifesaver for simplifying execution. I use kwargs in some places to declutter function parameters, but it's not consistent across all the code. 

I'm still using the code on a daily basis and learning new things as I track the outputs and results. I'll keep updating the funds.json file in this repo, and still want to revisit the whole ivol_forecasting system eventually. 

Logan


