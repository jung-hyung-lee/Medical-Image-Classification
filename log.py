import logging
from pathlib import Path
from datetime import datetime

def setupLogging(log_path):
    '''
    Take in a log directory and write all logs to this directory
    Usage:
        - call setupLogging("your_log_path") in the __main__=__name__ section
        - every time you want to print to console and log it into a file use:
            logging.info("Your message") 
    '''
    Path(log_path).mkdir(parents=True, exist_ok=True)    
    logFilePath = f"{log_path}/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
    logging.basicConfig(level = logging.INFO,
                        filename=logFilePath,
                        filemode='a',
                        format='%(asctime)s ::  %(filename)-12s :: %(funcName)-20s :: %(levelname)-8s :: %(lineno)5d :: %(message)s',
                        datefmt="%Y/%m/%d %H:%M:%S%z")

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(funcName)-20s :: %(levelname)-8s :: %(lineno)5d :: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)