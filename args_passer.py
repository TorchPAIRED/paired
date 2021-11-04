
args_global = None
def set_passed_args(args):
    global args_global
    args_global = args

def get_passed_args():
    return args_global