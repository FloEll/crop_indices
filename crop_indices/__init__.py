# -*- coding: utf-8 -*-
"""
First version created on Wed Jan 20 14:17:56 2021
@author: Florian Ellsäßer
Licence: This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""

# this is a statement that is printed to check weather the package has been 
# imported correctly
print(f'Invoking __init__.py for {__name__}')

# define a global variable
n_days_of_year = 365

# this helps to import all modules of the package with *
__all__ = [
        'indices',
        'utils',
        'spei_utils'
        ]