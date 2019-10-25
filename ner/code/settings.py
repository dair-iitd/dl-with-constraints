cuda = 0
def set_settings(params):
    global cuda
    cuda = params['cuda_device'] != -1
