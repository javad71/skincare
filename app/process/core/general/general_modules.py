import os
import glob


def truncate_content(folder):
    try:
        files = glob.glob(folder+'*')
        for f in files:
            os.remove(f)
        return True
    except:
        return False