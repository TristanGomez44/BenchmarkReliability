import os
import numpy as np
import sqlite3 
from scipy.stats._resampling import _bootstrap_iv,rng_integers,_percentile_of_score,ndtri,ndtr,BootstrapResult,ConfidenceInterval

import warnings
from scipy.stats._warnings_errors import DegenerateDataWarning

from krippendorf.utils import krippendorff_alpha_paralel,krippendorff_alpha_bootstrap
from metrics.faithfulness_metrics import get_sub_single_step_metric_list,get_sub_metric_list,get_metrics_to_minimize,get_cumulative_suff_list

from post_hoc_expl import utils as post_hoc_utils

def preprocc_matrix(metric_values_matrix,metric):

    if metric == "IIC":
        metric_values_matrix = metric_values_matrix.astype("bool")

    return metric_values_matrix

def fmt_value_str(value_str):

    value_list = value_str.split(";")

    if value_list[0].startswith("shape="):
        shape = value_list[0].replace("shape=(","").replace(")","").replace(",","").split(" ")
        shape = [int(size) for size in shape]

        value_list= value_list[1:]
    else:
        shape = (-1,)
    value_list = np.array(value_list).astype("float").reshape(shape)

    return value_list 

def fmt_metric_values(metric_values_list):
    matrix = []
    for i in range(len(metric_values_list)):
        matrix.append(fmt_value_str(metric_values_list[i]))
    metric_values_matrix = np.stack(matrix,axis=0)
    metric_values_matrix = metric_values_matrix.transpose(1,0)
    return metric_values_matrix

def convert_to_ordinal(metric_values_matrix,metric,metrics_to_minimize):
    if metric not in metrics_to_minimize and metric != "IIC":
        metric_values_matrix = -metric_values_matrix
    
    metric_values_matrix = metric_values_matrix.argsort(-1).argsort(-1)+1
    return metric_values_matrix

def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    """Bootstrap resample the sample."""
    n = len(sample)

    # bootstrap - each row is a random resample of original observations
    i = rng_integers(random_state, 0, n, (n,))

    resamples = np.array(sample)[i]
    return resamples

def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""

    # closely follows [1] 14.3 and 15.4 (Eq. 15.36)

    # calculate z0_hat
    theta_hat = np.asarray(statistic(*data))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)

    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_ji = []  # j is for sample of data, i is for jackknife resample
    theta_hat_i = []
    for i in range(len(data)):
        inds =[j for j in range(len(data))]
        inds.remove(i)
        inds = np.array(inds).astype("int")
        data = np.array(data)
        data_jackknife = data[inds]
        theta_hat_i.append(statistic(*data_jackknife)[0])
    theta_hat_ji.append(theta_hat_i)

    theta_hat_ji = [np.array(theta_hat_i)
                    for theta_hat_i in theta_hat_ji]

    n_j = [len(theta_hat_i) for theta_hat_i in theta_hat_ji]

    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True)
                       for theta_hat_i in theta_hat_ji]

    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i)
            for theta_hat_dot, theta_hat_i, n
            in zip(theta_hat_j_dot, theta_hat_ji, n_j)]

    nums = [(U_i**3).sum(axis=-1)/n**3 for U_i, n in zip(U_ji, n_j)]
    dens = [(U_i**2).sum(axis=-1)/n**2 for U_i, n in zip(U_ji, n_j)]
    a_hat = 1/6 * sum(nums) / sum(dens)**(3/2)

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2, a_hat  # return a_hat for testing

def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]

    alpha = np.broadcast_to(alpha[0], shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            # e.g. when bootstrap distribution has only one unique element
            msg = (
                "The BCa confidence interval cannot be calculated."
                " This problem is known to occur when the distribution"
                " is degenerate or the statistic is np.min."
            )
            warnings.warn(DegenerateDataWarning(msg))
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              method='BCa', bootstrap_result=None, random_state=None):
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, n_resamples, batch, method,
                         bootstrap_result, random_state)
    data, statistic, vectorized, paired, axis, confidence_level = args[:6]
    n_resamples, batch, method, bootstrap_result, random_state = args[6:]

    theta_hat_b = ([] if bootstrap_result is None
                   else [bootstrap_result.bootstrap_distribution])

    batch_nominal = batch or n_resamples or 1

    for k in range(0, n_resamples):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples

        resampled_data = _bootstrap_resample(data, n_resamples=batch_actual,
                                        random_state=random_state)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = (1 - confidence_level)/2
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1)),theta_hat_b

def get_background_func(background):
    if background is None:
        background_func = lambda x:"blur" if x in ["Insertion","IAUC","IC","INSERTION_VAL_RATE","IC-nc"] else "black"
    else:
        background_func = lambda x:background
    return background_func 

def keep_required_explanations(output,post_hoc_group,post_hoc_col_ind):
    accepted_methods = getattr(post_hoc_utils,"get_"+post_hoc_group+"_methods")()
    filtered_output = []
    for i in range(len(output)):
        if output[i][post_hoc_col_ind] in accepted_methods:
            filtered_output.append(output[i])
    return filtered_output
  
def krippendorf(metric_values_matrix_alpha,exp_id,metric,cumulative_suff,csv_krippen,array_krippen,array_krippen_err,metr_ind,krippendorf_sample_nb):

    alpha = krippendorff_alpha_paralel(metric_values_matrix_alpha)
    rng = np.random.default_rng(0)
    print(metric_values_matrix_alpha.shape)
    res,alpha_values = bootstrap(metric_values_matrix_alpha, krippendorff_alpha_bootstrap, confidence_level=0.99,random_state=rng,method="bca" ,vectorized=True,n_resamples=krippendorf_sample_nb)
    confinterv= res.confidence_interval
    csv_krippen += ","+str(alpha)+" ("+str(confinterv.low)+" "+str(confinterv.high)+")"

    array_krippen[1*(cumulative_suff == "-nc"),metr_ind] = alpha
    array_krippen_err[1*(cumulative_suff == "-nc"),metr_ind] = np.array([confinterv.low,confinterv.high])

    return csv_krippen, array_krippen, array_krippen_err,alpha_values

def addArgs(argreader):
    argreader.parser.add_argument('--background', type=str)
    argreader.parser.add_argument('--ordinal_metric', action="store_true")
    argreader.parser.add_argument('--krippendorf_sample_nb',type=int,default=5000)
    argreader.parser.add_argument('--post_hoc_group',type=str,default="all")
    return argreader

def compute_krippendorf_alpha(model_id,post_hoc_group,krippendorf_sample_nb,args,ordinal_metric=True,background=None,output_dir="../"):

    exp_id = args.exp_id

    single_step_metrics = get_sub_single_step_metric_list()
    metrics_to_minimize = get_metrics_to_minimize()
    metrics = get_sub_metric_list(include_noncumulative=False)

    background_func = get_background_func(background)

    db_path = f"{output_dir}/results/{exp_id}/saliency_metrics.db"
    con = sqlite3.connect(db_path) 
    cur = con.cursor()

    csv_krippen = "cumulative,"+ ",".join(metrics) + "\n"
    array_krippen=np.zeros((2,len(metrics)))
    array_krippen_err=np.zeros((2,len(metrics),2))

    explanation_names_list = []

    filename_suff = f"{model_id}_b{background}"

    outpath = f"{args.output_dir}/results/{exp_id}/krippendorff_alpha_values_list_{filename_suff}_{post_hoc_group}.csv"

    if not os.path.exists(outpath):
        alpha_values_list = []

        for cumulative_suff in get_cumulative_suff_list():

            csv_krippen += "False" if cumulative_suff == "-nc" else "True"

            for metr_ind,metric in enumerate(metrics):
                
                print(metric,cumulative_suff)

                if metric not in single_step_metrics or cumulative_suff=="": 
                    background = background_func(metric)
                    metric += cumulative_suff

                    query = f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}"'

                    output = cur.execute(query).fetchall()

                    output = list(filter(lambda x:x[0] != "",output))
                    
                    output = keep_required_explanations(output,post_hoc_group,post_hoc_col_ind=0)
                    explanation_names,metric_values_list = zip(*output)
                    
                    metric_values_matrix = fmt_metric_values(metric_values_list)
                    metric_values_matrix = preprocc_matrix(metric_values_matrix,metric)

                    if ordinal_metric:
                        metric_values_matrix_alpha = convert_to_ordinal(metric_values_matrix,metric,metrics_to_minimize)
                    else:
                        metric_values_matrix_alpha = metric_values_matrix

                    #Krippendorf's alpha (inter-rater reliabiilty) 
                    csv_krippen, array_krippen, array_krippen_err,alpha_values = krippendorf(metric_values_matrix_alpha,exp_id,metric,cumulative_suff,csv_krippen,array_krippen,array_krippen_err,metr_ind,krippendorf_sample_nb)

                    alpha_values_list.append((metric,alpha_values))
                    explanation_names_list.append(explanation_names)

            csv_krippen += "\n"

        with open(f"{args.output_dir}/results/{exp_id}/krippendorff_alpha_values_list_{filename_suff}_{post_hoc_group}.csv","w") as file:
            for row in alpha_values_list:
                print(row[0]+",",file=file,end="")
                values = ",".join([str(value) for value in row[1]])
                print(values,file=file)
            
        with open(f"{args.output_dir}/results/{exp_id}/krippendorff_alpha_{filename_suff}_{post_hoc_group}.csv","w") as file:
            print(csv_krippen,file=file)

        explanation_names_set = set(explanation_names_list)

        if len(explanation_names_set) != 1:
            print("Different sets of explanations methods were used:",explanation_names_set)
    else:
        print("Already computed krippendorff's alpha")