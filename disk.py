import numpy as np

def disk(r, center, shape):
  '''
	Makes a bool array containing an N-dimensional ellipsoid mask.

    Parameters
    ----------
    r : scalar or N-dimensional tuple     
        The radii of the ellipsoid, may be fractional.  If N-dimensional,
        elliptical radii are specified in each dimension.  If scalar,
        same radius applies to all dimensions. 
    center : tuple
        Gives the position of the center of the ellipsoid, may be
        fractional and of any dimension.  Note that if the desired
        "ellipsoid" is 1D, specifying (20) on the command line results
        in an int, not a tuple containing an int.  Say (20,) to force a
        tuple containing an int.
    shape :  tuple, int
        Gives the shape of the output array.  Must be integer and same
        length as center.

    Returns
    -------
    output : boolean array
    	This function returns a bool array containing an N-dimensional
    	ellipsoid (line segment, filled ellipse, ellipsoid, etc.).
    	The ellipsoid is centered at center and has the radii given by
    	r.  Shape specifies the shape.  The type is bool.  Array
    	values of 1 indicate that the center of a pixel is within the
    	given ellipsoid.  Pixel values of 0 indicate the opposite.
    	The center of each pixel is the integer position of that
    	pixel.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import disk
    >>> r      = (110, 150)
    >>> center = (230, 220)
    >>> shape  = (600, 500)
    >>> plt.imshow(disk.disk(r, center, shape), origin='lower', interpolation='nearest')
    >>> plt.gray()

    Revisions
    ---------
    2003-04-04 0.1  jh@oobleck.astro.cornell.edu Initial version.
    2007-11-25 0.2  jh@physics.ucf.edu           IDL->Python, made N-dim.
    2008-11-04 0.3  kstevenson@physics.ucf.edu   Updated docstring.
    2010-10-26 0.4  jh@physics.ucf.edu           Updated docstring.
    2016-10-27 0.5  jh@physics.ucf.edu           Made center and r not change.
    2017-10-31 0.6  jh@physics.ucf.edu           Sshape and rshape -> integers.
  '''
  idisk      = np.indices(shape, dtype=float)
  cctr       = np.asarray(center).copy()
  sshape     = np.ones(1 + cctr.size, dtype=int)
  sshape[0]  = cctr.size
  cctr.shape = sshape
  rr         = np.asarray(r).copy()
  rshape     = np.ones(1 + rr.size, dtype=int)
  rshape[0]  = rr.size
  rr.shape   = rshape

  return np.sum(((idisk - cctr)/rr)**2, axis=0) <= 1.
