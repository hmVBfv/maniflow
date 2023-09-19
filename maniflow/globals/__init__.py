import json
global GLOBALS


class __Globals:
    """
    A class to handle the global variables which are stored in maniflow/globals/data.json
    """
    def __init__(self):
        """
        This method initializes the class and immediately reads
        the stored values from the file
        """
        self.__data = self.readFromMemory("maniflow/globals/data.json")

    @staticmethod
    def readFromMemory(filename: str) -> dict:
        """
        A method that reads the stored values from the file given file
        :param filename: the path to the data file
        """
        print("read from memory...")
        with open(filename, "r") as file:
            content = file.read()
            file.close()
            return json.loads(content)

    def __getitem__(self, item: str):
        """
        A method that lets the user interact with an object of this class
        like a dictionary
        """
        return self.__data[item]

    def keys(self):
        """
        A method that lets the user interact with an object of this class
        like a dictionary
        """
        return self.__data.keys()

    @property
    def PRECISION(self):
        """
        A simple shorthand method that will return the global decimal
        precision
        """
        return self["precision"]


GLOBALS = __Globals()
