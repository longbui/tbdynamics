import numpy as np
import pymc as pm
import arviz as az
import multiprocessing as mp

from estival.model import BayesianCompartmentalModel
from estival.wrappers import pymc as epm
from estival.sampling import tools as esamp
from estival.wrappers import nevergrad as eng
from estival.utils.parallel import map_parallel
import nevergrad as ng

from tbdynamics.constants import BURN_IN, OPTI_DRAWS
from autumn.infrastructure.remote import springboard
from tbdynamics.model import build_model
from tbdynamics.calib_utils import get_bcm, get_all_priors, get_targets

def calibrate(out_path, draws, tune):
    bcm = get_bcm()
    def optimize_ng():
        opt = eng.optimize_model(bcm, budget=OPTI_DRAWS, opt_class=ng.optimizers.TwoPointsDE, obj_function=bcm.logposterior, num_workers=4)
        rec = opt.minimize(OPTI_DRAWS)
        return rec.value[1]

    opt_samples = optimize_ng()
    n_chains = 8
    n_samples = 100
    with pm.Model() as pm_model:
        variables = epm.use_model(bcm)
        idata_raw = pm.sample(step=[pm.DEMetropolisZ(variables)], draws=draws, tune=tune, cores=8, discard_tuned_samples=False, chains=n_chains, progressbar=False, initvals=opt_samples)
    idata_raw.to_netcdf(str(out_path / 'calib_full_out.nc'))
    
    burnt_idata = idata_raw.sel(draw=np.s_[BURN_IN:])
    idata_extract = az.extract(burnt_idata, num_samples=n_samples)
    
    spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm)
    spaghetti_res.results.to_hdf(str(out_path / 'results.hdf'), 'spaghetti')

    like_df = esamp.likelihood_extras_for_idata(idata_raw, bcm)
    like_df.to_hdf(str(out_path / 'results.hdf'), 'likelihood')

def run_calibration(bridge: springboard.task.TaskBridge, draws, tune):
    import multiprocessing as mp
    mp.set_start_method('forkserver')
    idata_raw = calibrate(bridge.out_path, draws, tune)
    bridge.logger.info('Calibration complete')
