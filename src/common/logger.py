class Logger:
    def __init__(self, exp_name, mode='w'):
        self.file = open('{}'.format(exp_name), mode)

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()