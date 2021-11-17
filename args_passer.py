
args_global = None
def set_passed_logdir(args):
    global args_global
    args_global = args

def get_passed_logdir():
    return args_global