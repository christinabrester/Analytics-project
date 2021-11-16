#-----------------------------------------------------------------------
#
# A class used to store the results of different models
#
#-----------------------------------------------------------------------

class Site:
    def __init__(self, **kwargs):
        self.__name = kwargs.get('name', [])
        self.__dates = kwargs.get('dates', [])
        self.__power_collection = kwargs.get('power_collection', [])
        self.__true_outcome = kwargs.get('true_output_name', 'power [W]')
        self.__models = kwargs.get('models', [])
        self.__test_id = kwargs.get('test_id', [])
          
    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        self.__name = name
        
    @property
    def dates(self):
        return self.__dates
    
    @dates.setter
    def dates(self, dates):
        self.__dates = dates
        
    @property
    def power_collection(self):
        return self.__power_collection
        
    @power_collection.setter
    def power_collection(self, df):
        self.__power_collection = df
        
    @property
    def true_outcome(self):
        return self.__true_outcome
    
    @true_outcome.setter
    def true_outcome(self, true_outcome):
        self.__true_outcome = true_outcome
    
    @property
    def models(self):
        return self.__models
    
    @models.setter
    def models(self, models):
        self.__models = models
        
    @property
    def test_id(self):
        return self.__test_id
    
    @test_id.setter
    def test_id(self, test_id):
        self.__test_id = test_id
