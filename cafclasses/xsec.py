from analysis_village.unfolding import wienersvd
import json
import os
import numpy as np
from sbnd.stats import stats
from sbnd.general import utils
import boost_histogram as bh
import uproot

class XSec:
    def __init__(self, response, xsec_unit, bins,
                 bin_widths=None, name='xsec', variable=''):
        """
        Parameters
        ----------
        response: array, response matrix
        sig_truth: array, signal distribution binned by true variable (now provided per unfold)
        xsec_unit: float, xsec unit
        bins: array, binnning of variable
        name: str, name of the xsec
        variable: str, variable of the xsec
        sel_background_reco: array, selection distribution binned by background reco variable (now provided per unfold)
        """
        self.response = response
        self.xsec_unit = xsec_unit
        self.bins = bins
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        if bin_widths is not None:
            self.bin_widths = bin_widths
        else:
            self.bin_widths = bins[1:] - bins[:-1]
        self.name = name
        self.variable = variable
        # Dictionary of unfold results keyed by a user-provided name.
        # Each entry stores configuration, sig_smear, sig_unfold, covariances, chi2, dof, pval, etc.
        self.unfold_results = {}
    def __str__(self):
        return f'XSec(name={self.name}, variable={self.variable}), keys={list(self.unfold_results.keys())}'
    def __repr__(self):
        return self.__str__()

    def get_cov(self, fraccov, sigma_tilde):
        return stats.covariance_from_fraccov(fraccov, sigma_tilde)

    def _init_unfold_record(self, name):
        """
        Ensure an unfold_results entry exists for a given name, with all keys present.
        This lets smear / unfold update only their relevant pieces without kludgy checks.
        """
        if name not in self.unfold_results:
            self.unfold_results[name] = {
                'C_type': None,
                'Norm_type': None,
                'fractional_cov': None,
                'input_cov': None,
                'unfold_cov': None,
                'AddSmear': None,
                'unfold': None,
                'sig_truth': None,
                'sel_background': None,
                'sigma_tilde': None,
                'sig_smear': None,
                'sig_unfold': None,
                'fracunc': None,
                'chi2': None,
                'dof': None,
                'pval': None,
            }
        return self.unfold_results[name]

    def _compute_wiener(
        self,
        cov,
        sel,
        sig_truth,
        sel_background=None,
        sigma_tilde=None,
        C_type=2,
        Norm_type=0.5,
        verbose=False,
        eps=1e-6,
        fractional_cov=False,
    ):
        """
        Core Wiener-SVD call shared by smear and unfold.
        Returns a dict with all raw and derived quantities.
        """
        if sel_background is not None:
            sigma_tilde = stats.compute_sigma_tilde(
                self.response,
                sig_truth,
                sel_background,
                self.xsec_unit,
            )
        elif sigma_tilde is None:
            raise ValueError('sigma_tilde is not set')
        if fractional_cov:
            cov = self.get_cov(cov, sigma_tilde)

        input_cov = np.array(cov)

        sig = sig_truth * self.xsec_unit
        sel = sel * self.xsec_unit
        # Replace any sig or sel that are 0 with eps times minimum value
        min_value = np.min(np.abs(sig[sig != 0]))
        sig = np.where(sig == 0, eps * min_value, sig)
        min_value = np.min(np.abs(sel[sel != 0]))
        sel = np.where(sel == 0, eps * min_value, sel)
        if verbose:
            print(f'sig: {sig}')
            print(f'sel: {sel}')
            print(f'sig@response: {sig @ self.response}')
            print(f'sig@response - sel: {sig @ self.response - sel}')
            print(f'sqrt(cov diag): {np.sqrt(np.diag(input_cov))}')
            print(f'sqrt(cov diag)/sel*100: {(np.sqrt(np.diag(input_cov)) / sel) * 100}')
            print(f'sigma_tilde: {sigma_tilde}')
            print(f'xsec_unit: {self.xsec_unit}')
            print(f'bins: {self.bins}')

        unfold_dict = wienersvd.WienerSVD(
            self.response,
            sig,
            sel,
            input_cov,
            C_type,
            Norm_type
        )

        sig_smear = unfold_dict['AddSmear'] @ sig / self.bin_widths
        sig_unfold = unfold_dict['unfold'] / self.bin_widths
        # The bin widths need to be included since the covariance does not include them
        unfold_cov = unfold_dict['UnfoldCov']
        fracunc = stats.get_fractional_uncertainty(
            sig_unfold * self.bin_widths,
            unfold_cov,
        )

        sig_unfold_bw = sig_unfold * self.bin_widths
        sig_smear_bw = sig_smear * self.bin_widths
        if verbose:
            print(f'sig_unfold_bw: {sig_unfold_bw}')
            print(f'sig_smear_bw: {sig_smear_bw}')
            print(f'unfold cov diag: {np.diag(unfold_cov)}')
        chi2, dof, pval = stats.calc_chi2(sig_unfold_bw, sig_smear_bw, unfold_cov)
        chi2_dof_pval_str = utils.get_chi2_dof_pval_str(chi2, dof, pval)

        # Optional per-test sigma_tilde if a background is provided
        sigma_tilde = None
        if sel_background is not None:
            sigma_tilde = stats.compute_sigma_tilde(
                self.response,
                sig_truth,
                sel_background,
                self.xsec_unit,
            )

        return {
            'C_type': C_type,
            'Norm_type': Norm_type,
            'fractional_cov': fractional_cov,
            'input_cov': input_cov,
            'unfold_cov': unfold_cov,
            'AddSmear': unfold_dict['AddSmear'],
            'unfold': unfold_dict['unfold'],
            'sig_truth': sig_truth,
            'sel_background': sel_background,
            'sigma_tilde': sigma_tilde,
            'sig_smear': sig_smear,
            'sig_unfold': sig_unfold,
            'fracunc': fracunc,
            'chi2': chi2,
            'dof': dof,
            'pval': pval,
            'chi2_dof_pval_str': chi2_dof_pval_str,
        }

    def unfold(
        self,
        cov,
        sel,
        sig_truth,
        sel_background=None,
        sigma_tilde=None,
        name=None,
        **kwargs,
    ):
        """
        Unfold the xsec using Wiener-SVD and store unfolded results in a named entry.

        Parameters
        ----------
        cov: array, covariance matrix (or fractional covariance if fractional_cov is True)
        sel: array, selection distribution binned by reco variable
        sig_truth: array, signal truth spectrum for this test
        sel_background: array, background spectrum used to compute sigma_tilde
        name: str, label for this unfold configuration (defaults to self.name)
        **kwargs: additional keyword arguments for _compute_wiener
        """
        if name is None:
            name = self.name
        record = self._init_unfold_record(name)
        core = self._compute_wiener(
            cov=cov,
            sel=sel,
            sig_truth=sig_truth,
            sel_background=sel_background,
            **kwargs,
        )
        # Update all fields relevant to a full unfolding
        record.update(core)

    def save(self, save_dir, metadata_dir='metadata_xsec'):
        """
        Save the XSec metadata and any stored unfold results.

        Parameters
        ----------
        save_dir: str, directory to save into
        metadata_dir: str, subdirectory for metadata and unfold results
        """
        os.makedirs(save_dir, exist_ok=True)
        meta_dir = os.path.join(save_dir, metadata_dir)
        os.makedirs(meta_dir, exist_ok=True)
        # Core metadata arrays (truth and backgrounds are now per-unfold and live in unfold_results)
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_bins.csv'), self.bins)
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_response.csv'), self.response)
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_bin_widths.csv'), self.bin_widths)
        if getattr(self, 'sigma_tilde', None) is not None:
            np.savetxt(os.path.join(meta_dir, f'{self.variable}_sigma_tilde.csv'), self.sigma_tilde)
        # Scalars and strings
        np.savetxt(os.path.join(meta_dir, f'{self.variable}_xsec_unit.csv'), np.array([self.xsec_unit]))
        meta = {'name': self.name, 'variable': self.variable}
        with open(os.path.join(meta_dir, f'{self.variable}_metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        # Unfold results: save each named configuration in its own subfolder, with
        # filenames prefixed by the variable name (to mirror Systematics.save layout).
        if self.unfold_results:
            for key, record in self.unfold_results.items():
                subfolder = os.path.join(save_dir, key)
                os.makedirs(subfolder, exist_ok=True)
                filename_prefix = f'{self.variable}_'
                for k, val in record.items():
                    if val is None:
                        continue
                    # Skip empty lists / arrays
                    if isinstance(val, (list, np.ndarray)) and len(val) == 0:
                        continue
                    path_base = os.path.join(subfolder, f'{filename_prefix}{k}')
                    # Arrays or lists -> CSV
                    if isinstance(val, (np.ndarray, list)):
                        arr = np.array(val)
                        # 0D arrays become scalars, handle below
                        if arr.ndim == 0:
                            with open(path_base + '.txt', 'w') as f:
                                f.write(str(arr.item()))
                        else:
                            np.savetxt(path_base + '.csv', arr)
                    # Strings
                    elif isinstance(val, str):
                        with open(path_base + '.txt', 'w') as f:
                            f.write(val)
                    # Scalars (int, float, numpy scalar)
                    elif isinstance(val, (int, float, np.number)):
                        with open(path_base + '.txt', 'w') as f:
                            f.write(str(val))
                    else:
                        # Best-effort string representation for anything odd
                        with open(path_base + '.txt', 'w') as f:
                            f.write(str(val))

    def to_nuisance(self, save_dir, name=None, filename_suffix='xsec', scale=1.):
        """
        Write covariance matrix, unfolded spectrum, and smearing matrix to a ROOT file
        in uboone data-release style: covariance_matrix (TH2D), smearing_matrix (TH2D),
        xsec_data (TH1D).

        Parameters
        ----------
        save_dir : str or path-like
            Output directory.
        name : str, optional
            Key in unfold_results to use. If None, uses self.name; if that config is
            missing, uses the first available key.
        filename_suffix : str, optional
            Suffix for the filename.
        scale : float, optional
            Scale factor for the unfolded spectrum, covariance matrix, and smearing matrix.
        """
        key = name if name is not None else self.name
        if key not in self.unfold_results:
            raise RuntimeError(f'unfold config "{key}" not found; run unfold() for this config first')
        save_dir = os.path.join(save_dir,key)
        filename = f'{self.variable}_{filename_suffix}.root'
        record = self.unfold_results[key]
        unfold = record.get('unfold')
        unfold_cov = record.get('unfold_cov')
        add_smear = record.get('AddSmear')
        if unfold is None or unfold_cov is None or add_smear is None:
            raise RuntimeError(
                f'unfold config "{key}" missing unfold, unfold_cov, or AddSmear; '
                'run unfold() for this config first'
            )
        n = len(unfold)
        bins = np.asarray(self.bins, dtype=np.float64)
        if len(bins) != n + 1:
            raise RuntimeError(
                f'bins length {len(bins)} != n+1 ({n}+1); check XSec bins vs unfold size'
            )
        axis = bh.axis.Variable(bins)
        h_xsec = bh.Histogram(axis, storage=bh.storage.Double())
        h_xsec.view(flow=True)[1:-1] = np.asarray(unfold/self.bin_widths, dtype=np.float64) * scale
        h_cov = bh.Histogram(axis, axis, storage=bh.storage.Double())
        h_cov.view(flow=True)[1:-1, 1:-1] = np.asarray(unfold_cov, dtype=np.float64) * scale
        h_smear = bh.Histogram(axis, axis, storage=bh.storage.Double())
        h_smear.view(flow=True)[1:-1, 1:-1] = np.asarray(add_smear, dtype=np.float64) * scale
        with uproot.recreate(os.path.join(save_dir,filename)) as f:
            # TODO: Replace this with uBooNE data-release style
            # f['covariance_matrix'] = h_cov
            # f['smearing_matrix'] = h_smear
            # f['xsec_data'] = h_xsec
            # uBooNE data-release style
            f['covariance'] = h_cov
            f['smearing_Ac'] = h_smear
            f['data_xsec'] = h_xsec

    @classmethod
    def from_saved(cls, load_dir, variable, metadata_dir='metadata_xsec'):
        """
        Reconstruct an XSec object and any stored unfold results from disk.

        Parameters
        ----------
        load_dir: str, directory to load from
        variable: str
            Variable name / type to load. This selects which variable-specific
            metadata files (by prefix) and unfold-result files to read.
        metadata_dir: str, subdirectory for metadata and unfold results
        """
        meta_dir = os.path.join(load_dir, metadata_dir)
        if not os.path.exists(meta_dir):
            raise ValueError(f'Metadata directory not found in {load_dir}')
        bins = np.loadtxt(os.path.join(meta_dir, f'{variable}_bins.csv'))
        response = np.loadtxt(os.path.join(meta_dir, f'{variable}_response.csv'))
        bin_widths = np.loadtxt(os.path.join(meta_dir, f'{variable}_bin_widths.csv'))
        sigma_tilde_path = os.path.join(meta_dir, f'{variable}_sigma_tilde.csv')
        sigma_tilde = np.loadtxt(sigma_tilde_path) if os.path.exists(sigma_tilde_path) else None
        xsec_unit_arr = np.loadtxt(os.path.join(meta_dir, f'{variable}_xsec_unit.csv'))
        if np.ndim(xsec_unit_arr) == 0:
            xsec_unit = float(xsec_unit_arr)
        else:
            xsec_unit = float(xsec_unit_arr[0])
        meta_path = os.path.join(meta_dir, f'{variable}_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            name = meta.get('name', 'xsec')
        else:
            name = 'xsec'
        instance = cls(
            response=response,
            xsec_unit=xsec_unit,
            bins=bins,
            bin_widths=bin_widths,
            name=name,
            variable=variable,
        )
        if sigma_tilde is not None:
            instance.sigma_tilde = sigma_tilde
        # Load unfold results: each subfolder (except metadata_dir_name) corresponds to a key.
        # Only files whose names start with the chosen variable prefix are loaded,
        # so different variable types can safely coexist in the same directory.
        instance.unfold_results = {}
        filename_prefix = f'{instance.variable}_'
        for entry in os.listdir(load_dir):
            subfolder_path = os.path.join(load_dir, entry)
            if not os.path.isdir(subfolder_path):
                continue
            # Skip metadata directory for this variable
            if entry == metadata_dir:
                continue
            key = entry
            record = instance._init_unfold_record(key)
            for fname in os.listdir(subfolder_path):
                fpath = os.path.join(subfolder_path, fname)
                if not os.path.isfile(fpath):
                    continue
                # Only consider files that start with the variable prefix
                if not fname.startswith(filename_prefix):
                    continue
                # Strip prefix and extension to get the record field name
                base = fname.replace('.csv', '').replace('.txt', '')
                if not base.startswith(filename_prefix):
                    continue
                field_name = base[len(filename_prefix):]
                # CSV -> numpy array
                if fname.endswith('.csv'):
                    arr = np.loadtxt(fpath)
                    record[field_name] = arr
                # TXT -> scalar or string
                elif fname.endswith('.txt'):
                    with open(fpath, 'r') as f:
                        txt = f.read().strip()
                    # Try to parse as int / float; fall back to string
                    try:
                        num = float(txt)
                        if num.is_integer():
                            record[field_name] = int(num)
                        else:
                            record[field_name] = num
                    except ValueError:
                        record[field_name] = txt
            instance.unfold_results[key] = record
        return instance

    def load(self, load_dir, variable, metadata_dir='metadata_xsec'):
        """
        Load unfold results into an already constructed XSec object.

        Parameters
        ----------
        load_dir: str, directory to load from
        variable: str
            Variable name / type to load. This selects which variable-specific
            unfold-result files to read (by filename prefix).
        metadata_dir: str, subdirectory where shared metadata lives
        """
        meta_dir = os.path.join(load_dir, metadata_dir)
        if not os.path.exists(meta_dir):
            return

        filename_prefix = f'{variable}_'

        for entry in os.listdir(load_dir):
            subfolder_path = os.path.join(load_dir, entry)
            if not os.path.isdir(subfolder_path):
                continue
            if entry == metadata_dir:
                continue
            key = entry
            record = self._init_unfold_record(key)
            for fname in os.listdir(subfolder_path):
                fpath = os.path.join(subfolder_path, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.startswith(filename_prefix):
                    continue
                base = fname.replace('.csv', '').replace('.txt', '')
                if not base.startswith(filename_prefix):
                    continue
                field_name = base[len(filename_prefix):]
                if fname.endswith('.csv'):
                    arr = np.loadtxt(fpath)
                    record[field_name] = arr
                elif fname.endswith('.txt'):
                    with open(fpath, 'r') as f:
                        txt = f.read().strip()
                    try:
                        num = float(txt)
                        if num.is_integer():
                            record[field_name] = int(num)
                        else:
                            record[field_name] = num
                    except ValueError:
                        record[field_name] = txt
            self.unfold_results[key] = record

    @staticmethod
    def _to_serializable(obj):
        """
        Recursively convert numpy arrays in an object to lists for JSON storage.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: XSec._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [XSec._to_serializable(v) for v in obj]
        return obj

    @staticmethod
    def _from_serializable(obj):
        """
        Recursively convert JSON loaded structures back to numpy arrays
        where appropriate (lists of numbers or nested lists).
        """
        if isinstance(obj, dict):
            return {k: XSec._from_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            # Heuristic: convert lists of numbers or nested lists to numpy arrays
            if not obj:
                return np.array([])
            if all(isinstance(v, (int, float)) for v in obj):
                return np.array(obj)
            if all(isinstance(v, list) for v in obj):
                return np.array([XSec._from_serializable(v) for v in obj])
            return [XSec._from_serializable(v) for v in obj]
        return obj