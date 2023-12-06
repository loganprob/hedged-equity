_Framework for comparing underlying option exposures of exchange-traded "Buffer" equity funds; running forward-looking simulations and pricing the option holdings under those hypothetical conditions, as an analogy to fund performance_



## Code is currently... Broken :(
12/5/24
- after months of uninterupted execution, the week after I put it on github, the CBOE website changed how option data is delivered/display, naturally
- previously, when you made a basic GET requested to the site, buried within the HTML they very kindly put the entire option chain in JSON syntax... _see the get_spy_option_chain() function_
- that data now appears to be gone, and their site isn't broken which I assume they don't need the data sitting out like that
- need to explore alternatives (Schwab's retail website get's the whole option chain delivered in json upon loading, but that's annoying to login and pull manually)
- was working on version 2 in the midst of this, which is now paused
