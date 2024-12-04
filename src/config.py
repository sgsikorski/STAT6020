class Config:
    DEBUG = False
    EVALUATE = False
    USE_SKLEARN = False
    OUTPUT_SUFFIX = ""
    SAVE_FIGS = False

    @staticmethod
    def SetConfig(args):
        for arg in args:
            if arg == "-d":
                Config.DEBUG = True
            elif arg == "-s" or arg == "--sklearn":
                Config.USE_SKLEARN = True
            elif arg == "--save":
                Config.SAVE_FIGS = True
            elif arg == "-e" or arg == "--eval":
                Config.EVALUATE = True
            elif arg.startswith("-o"):
                Config.OUTPUT_SUFFIX = arg[2:]
            else:
                print(f"Unknown argument: {arg}")
