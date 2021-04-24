"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
# from . import cmudict
from .korean import ALL_SYMBOLS, PAD, EOS


# For english
en_symbols = PAD+EOS+'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '  #<-For deployment(Because korean ALL_SYMBOLS follow this convention)

symbols = ALL_SYMBOLS # for korean
# 
# # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# #_arpabet = ["@' + s for s in cmudict.valid_symbols]
# 
# # Export all symbols:
# symbols = [PAD, EOS] + list(_characters) #+ _arpabet
