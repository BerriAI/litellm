"""Provides a wrapper for GEOS types and functions.

Note: GEOS functions in Cython must be called using the get_geos_handle context
manager.

Examples
--------
    with get_geos_handle() as geos_handle:
        SomeGEOSFunc(geos_handle, ...<other params>)

"""

cdef extern from "geos_c.h":
    # Types
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSCoordSequence
    ctypedef void (*GEOSMessageHandler_r)(const char *message, void *userdata)

    # GEOS Context & Messaging
    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil
    void GEOSContext_setErrorMessageHandler_r(GEOSContextHandle_t handle, GEOSMessageHandler_r ef, void* userData) nogil
    void GEOSContext_setNoticeMessageHandler_r(GEOSContextHandle_t handle, GEOSMessageHandler_r nf, void* userData) nogil

    # Geometry functions
    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n) nogil
    const GEOSGeometry* GEOSGetExteriorRing_r(GEOSContextHandle_t handle, const GEOSGeometry* g) nogil
    const GEOSGeometry* GEOSGetInteriorRingN_r(GEOSContextHandle_t handle, const GEOSGeometry* g, int n) nogil
    int GEOSGeomTypeId_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil

    # Geometry creation / destruction
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle, const GEOSGeometry* g) nogil
    GEOSGeometry* GEOSGeom_createPoint_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    GEOSGeometry* GEOSGeom_createLineString_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    GEOSGeometry* GEOSGeom_createLinearRing_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    GEOSGeometry* GEOSGeom_createEmptyPolygon_r(GEOSContextHandle_t handle) nogil
    GEOSGeometry* GEOSGeom_createPolygon_r(GEOSContextHandle_t handle, GEOSGeometry* shell, GEOSGeometry** holes, unsigned int nholes) nogil
    GEOSGeometry* GEOSGeom_createCollection_r(GEOSContextHandle_t handle, int type, GEOSGeometry** geoms, unsigned int ngeoms) nogil
    void GEOSGeom_destroy_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil

    # Coordinate sequences
    const GEOSCoordSequence* GEOSGeom_getCoordSeq_r(GEOSContextHandle_t handle, const GEOSGeometry* g)
    GEOSCoordSequence* GEOSCoordSeq_clone_r(GEOSContextHandle_t handle, const GEOSCoordSequence* s)
    GEOSCoordSequence* GEOSCoordSeq_create_r(GEOSContextHandle_t handle, unsigned int size, unsigned int dims) nogil
    void GEOSCoordSeq_destroy_r(GEOSContextHandle_t handle, GEOSCoordSequence* s) nogil
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil
    int GEOSCoordSeq_setZ_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int idx, double val) nogil
    int GEOSCoordSeq_getSize_r(GEOSContextHandle_t handle, GEOSCoordSequence* s, unsigned int* size) nogil


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef char* last_error
    cdef char* last_warning
    cdef GEOSContextHandle_t __enter__(self)
