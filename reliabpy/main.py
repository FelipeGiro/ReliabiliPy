from reliabpy.examples.offshore_wind_turbine import Simple

def run():
    model = Simple()
    model.mount_model()
    model.run_one_episode()

if __name__ == '__main__':
    import cProfile, pstats
    import subprocess
    import time
    model = Simple()
    model.mount_model()
    profiler = cProfile.Profile()
    start = time.time()
    profiler.enable()
    model.run_one_episode()
    profiler.disable()
    print("Elapsed time for one episode", time.time() - start, 's')
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.sort_stats('tottime')
    stats.reverse_order()
    # stats.print_stats()
    # stats.dump_stats('profile.dat')
    # subprocess.call(r"snakeviz profile.dat")