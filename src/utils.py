def calculate_fps(time_elapsed):
    # convert to seconds (from milliseconds)
    t = time_elapsed
    # save to attribute
    return round(1 / t)