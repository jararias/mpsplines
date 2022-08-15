
import numpy as np
import pandas as pd

import csrad
import sunpos


times_utc = pd.date_range('2022-06-15', periods=24*60, freq='T')
solpos = sunpos.sites(times_utc, latitude=37.5, longitude=-3.5)
sim = csrad.REST2(cosz=solpos.cosz, ecf=solpos.ecf)
eth = np.where(solpos.cosz < 0, 0., 1361.*solpos.ecf*solpos.cosz)
df = pd.DataFrame(
    data={'ghi_W/m2': sim['ghi'], 'eth_W/m2': eth},
    index=times_utc
)
df.index.name = 'times_utc'
df.to_csv('ghics_profile.csv')
