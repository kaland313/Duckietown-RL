from unittest import TestCase
from duckietown_utils.utils import recursive_dict_update

class Test(TestCase):
    def test_recursive_dict_update(self):
        target = {'k0': 'v0',
                  'k1': 'v1',
                  'k2': {'k20': 'v20',
                         'k21': 'v21',
                         'k22': 'v22'
                         }
                  }
        update_dict = {'k1': '_v1',
                       'k2': {'k21': '_v21',
                              'k22': '_v22'
                              }
                       }
        self.assertEqual(update_dict, recursive_dict_update({}, update_dict))
        self.assertEqual(target, recursive_dict_update(target, {}))
        self.assertEqual({'k0': 'v0',
                          'k1': '_v1',
                          'k2': {'k20': 'v20',
                                 'k21': '_v21',
                                 'k22': '_v22'
                                 }
                          },
                         recursive_dict_update(target, update_dict))
