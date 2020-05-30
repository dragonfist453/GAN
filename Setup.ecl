IMPORT Python3 as Python;

String install() := EMBED(Python)
    import pip
    def install(package):
        if hasattr(pip,'main'):
            pip.main(['install', package])
        else:
            pip._internal(['install', package])

    install('tensorflow')
    install('opencv-python')
    install('matplotlib')
    return 'success'
ENDEMBED;

OUTPUT(install());