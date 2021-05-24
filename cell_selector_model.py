from shared_variables import *

class CellSelectorModel:
    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.video_session = None
        self.fileName = None
        self.fileContent = ""

    def create_video_session(self, filename):
        from video_session import VideoSession
        try:
            f = open(filename, 'r')
            f.close()
            self.video_session = VideoSession(filename)
            return self.video_session
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' does not exist or is not readable")


