
from admetnet.Utils.Imports import *

def nested_dataclass(*args, **kwargs):
   """
   Decorator to wrap original __init__ for dataclass

   usage:
         @dataclass 
         class A:
            a: int
            b: float
   
         @nested_dataclass
         class B:
            c: str
            d: A
   """
   def wrapper(check_class):
      check_class = dataclass(check_class, **kwargs)
      o_init = check_class.__init__
        
      def __init__(self, *args, **kwargs):

         for name, value in kwargs.items():
            ft = check_class.__annotations__.get(name, None)
            if is_dataclass(ft) and isinstance(value, dict):
               obj = ft(**value)
               kwargs[name]= obj
            o_init(self, *args, **kwargs)

         check_class.__init__=__init__
        
         return check_class
    
      return wrapper(args[0]) if args else wrapper


class Debugger(object):
   """
   Decorator for debugging functions
   """

   _level     = 1
   _debugfile = "DeepModeller.debug"

   def __init__(self, func):
      self.func = func

   def __call__(self, *args, **kwargs):
      if self._level == 1:
         with open(self._debugfile, "a") as opened_file:
            opened_file.write("Calling : %r\n" % self.func.__code__)

      elif self._level == 2:
         return_value = self.func(*args,**kwargs)

         with open(self._debugfile, "a") as opened_file:
            opened_file.write("Calling : %r\n" % self.func.__code__)
            opened_file.write("\targs, kwargs: %r , %r\n" % (args, kwargs))
            opened_file.write("\treturn_value : %r\n" % return_value)

      elif self._level == 3:
         start        = time.time()
         return_value = self.func(*args,**kwargs)
         duration     = time.time() - start

         with open(self._debugfile, "a") as opened_file:
            opened_file.write("Calling : %r\n" % self.func.__code__)
            opened_file.write("\targs, kwargs : %r , %r\n" % (args, kwargs))
            opened_file.write("\treturn_value : %r\n" % return_value)
            opened_file.write("\tduration     : %r\n" % duration)
      else:
         pass

      return self.func(*args)
