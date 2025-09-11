"""
Provides a wrapper for the shapely.lib C API for use in Cython.
Internally, the shapely C extension uses a PyCapsule to provide run-time access
to function pointers within the C API.

To use these functions, you must first call the following function in each Cython module:
`import_shapely_c_api()`

This uses a macro to dynamically load the functions from pointers in the PyCapsule.
Each C function in shapely.lib exposed in the C API must be specially-wrapped to enable
this capability.

Segfaults will occur if the C API is not imported properly.
"""

cimport numpy as np
from cpython.ref cimport PyObject

from shapely._geos cimport GEOSContextHandle_t, GEOSCoordSequence, GEOSGeometry


cdef extern from "c_api.h":
    cdef enum ShapelyErrorCode:
        PGERR_SUCCESS,
        PGERR_NOT_A_GEOMETRY,
        PGERR_GEOS_EXCEPTION,
        PGERR_NO_MALLOC,
        PGERR_GEOMETRY_TYPE,
        PGERR_MULTIPOINT_WITH_POINT_EMPTY,
        PGERR_COORD_OUT_OF_BOUNDS,
        PGERR_EMPTY_GEOMETRY,
        PGERR_GEOJSON_EMPTY_POINT,
        PGERR_LINEARRING_NCOORDS,
        PGERR_NAN_COORD,
        PGWARN_INVALID_WKB,
        PGWARN_INVALID_WKT,
        PGWARN_INVALID_GEOJSON,
        PGERR_PYSIGNAL

    cpdef enum ShapelyHandleNan:
        SHAPELY_HANDLE_NAN_ALLOW,
        SHAPELY_HANDLE_NAN_SKIP,
        SHAPELY_HANDLE_NANS_ERROR

    # shapely.lib C API loader; returns -1 on error
    # MUST be called before calling other C API functions
    int import_shapely_c_api() except -1

    # C functions provided by the shapely.lib C API
    # Note: GeometryObjects are always managed as Python objects
    # in Cython to avoid memory leaks, not PyObject* (even though
    # they are declared that way in the header file).
    object PyGEOS_CreateGeometry(GEOSGeometry *ptr, GEOSContextHandle_t ctx)
    char PyGEOS_GetGEOSGeometry(PyObject *obj, GEOSGeometry **out) nogil
    int PyGEOS_CoordSeq_FromBuffer(
        GEOSContextHandle_t ctx, const double* buf, unsigned int size,
        unsigned int dims, char is_ring, int handle_nan,
        GEOSCoordSequence** coord_seq) nogil
