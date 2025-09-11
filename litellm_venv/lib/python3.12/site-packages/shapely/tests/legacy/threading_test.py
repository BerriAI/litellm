import threading
from binascii import b2a_hex


def main():
    num_threads = 10
    use_threads = True

    if not use_threads:
        # Run core code
        runShapelyBuilding()
    else:
        threads = [
            threading.Thread(target=runShapelyBuilding, name=str(i), args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def runShapelyBuilding(num):
    print(f"{num}: Running shapely tests on wkb")
    import shapely.geos

    print(f"{num} GEOS Handle: {shapely.geos.lgeos.geos_handle}")
    import shapely.wkb
    import shapely.wkt

    p = shapely.wkt.loads("POINT (0 0)")
    print(f"{num} WKT: {shapely.wkt.dumps(p)}")
    wkb = shapely.wkb.dumps(p)
    print(f"{num} WKB: {b2a_hex(wkb)}")

    for i in range(10):
        shapely.wkb.loads(wkb)

    print(f"{num} GEOS Handle: {shapely.geos.lgeos.geos_handle}")
    print(f"Done {num}")


if __name__ == "__main__":
    main()
