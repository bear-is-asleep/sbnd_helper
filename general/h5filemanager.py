"""
H5FileManager class to handle XRootD URLs for h5py/pandas without copying files.
"""
import os
import h5py
import fsspec
from contextlib import contextmanager


def convert_pnfs_to_xroot(path):
    """Convert /pnfs path to XRootD URL."""
    if path.startswith("/pnfs"):
        return path.replace("/pnfs", "root://fndcadoor.fnal.gov:1094/pnfs/fnal.gov/usr")
    return path


class H5FileManager:
    """
    Context manager for h5py.File that handles XRootD URLs without copying.
    Tries local /pnfs mount first, then falls back to XRootD via fsspec.
    """
    
    def __init__(self, path, mode='r', **kwargs):
        """
        Initialize H5FileManager.
        
        Parameters
        ----------
        path : str
            Path to HDF5 file (can be /pnfs, root://, or local path)
        mode : str
            File mode (default 'r' for read)
        **kwargs
            Additional arguments to pass to h5py.File
        """
        self.original_path = path
        self.path = convert_pnfs_to_xroot(path)
        self.mode = mode
        self.kwargs = kwargs
        self.h5file = None
        self._fsspec_context = None
        self._fobj = None
    
    def __enter__(self):
        """Enter context manager - open file and return h5py.File object."""
        # Try local /pnfs mount first (fastest!)
        if self.original_path.startswith("/pnfs") and os.path.exists(self.original_path):
            self.h5file = h5py.File(self.original_path, mode=self.mode, **self.kwargs)
            return self.h5file
        
        # For XRootD URLs, use fsspec to open directly
        if self.path.startswith("root://"):
            # Open fsspec file context and keep it open
            self._fsspec_context = fsspec.open(self.path, "rb")
            self._fobj = self._fsspec_context.__enter__()
            # Create h5py file with the file object - driver='fileobj' is required
            self.h5file = h5py.File(self._fobj, mode=self.mode, **self.kwargs, driver='fileobj')
            return self.h5file
        
        # Regular local file
        self.h5file = h5py.File(self.path, mode=self.mode, **self.kwargs)
        return self.h5file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - close h5py file and fsspec context if needed."""
        try:
            if self.h5file is not None:
                self.h5file.close()
        finally:
            # Close fsspec context if it was opened
            if self._fsspec_context is not None:
                self._fsspec_context.__exit__(exc_type, exc_val, exc_tb)
    
    def __getattr__(self, name):
        """Delegate attribute access to h5py.File object when file is open."""
        if self.h5file is not None:
            return getattr(self.h5file, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


@contextmanager
def xrootd_file(xrootd_path, cleanup=True, verbose=False):
    """
    Context manager to handle XRootD URLs for pandas read_hdf.
    Checks if /pnfs is mounted locally first - much faster!
    For h5py, use H5FileManager instead.
    
    Note: Pandas read_hdf requires a file path, not a file object, so XRootD URLs
    must be copied to a temp file. This is why it falls back to copying.
    """
    original_path = xrootd_path
    xrootd_path = convert_pnfs_to_xroot(xrootd_path)
    
    # If it was originally /pnfs, try using local mount first (fastest!)
    if original_path.startswith("/pnfs") and os.path.exists(original_path):
        if verbose:
            print(f"Using local /pnfs mount: {original_path}")
        yield original_path
    elif xrootd_path.startswith("root://"):
        # For pandas, we must copy since read_hdf needs a file path, not a file object
        # Use unique temp filename to avoid conflicts from previous runs
        import tempfile
        import subprocess
        import uuid
        temp_dir = tempfile.gettempdir()
        base_filename = os.path.basename(xrootd_path)
        # Add uuid to make filename unique
        temp_filename = f"{base_filename}_{uuid.uuid4().hex[:8]}"
        local_path = os.path.join(temp_dir, temp_filename)
        
        if verbose:
            print(f"Copying {xrootd_path} to {local_path}...")
        result = subprocess.run(["xrdcp", xrootd_path, local_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to copy from XRootD: {result.stderr}")
        if verbose:
            print(f"Copied successfully")
        
        try:
            yield local_path
        finally:
            if cleanup:
                try:
                    os.remove(local_path)
                    if verbose:
                        print(f"Cleaned up temp file: {local_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temp file: {e}")
    else:
        # Regular local file path
        yield xrootd_path
