class config_loader(object):
    def __init__(self,file_):
        data = None
        with open(file_,'r') as f:
            data = f.read()
        data = data.split("\n")
        for line in data:
	    if line == "": continue
            if line.startswith("#"): continue
            SS = "self." + line
            print SS
            exec(SS)
        
