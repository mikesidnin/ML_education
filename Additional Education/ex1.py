# np just short
import numpy as np

# add a new array
# and check it's type. result int32
new_array = np.array([1, 2, 3, 4])
print(new_array.dtype)

# new array with mixed types of elements
# dtype <U32 and each symbol automatically transformed to String
# DO NOT MIX TYPES IN ARRAY!
new_mixed_array = np.array([1, "2", 3.1, True, 5, 6, 7, 8, 9])
print(new_mixed_array.dtype)

# let's pick the first element of array
print(new_mixed_array[0])

# let's change the value of the second element
new_mixed_array[1] = 'Mike'
print(new_mixed_array)

# let's get a new array by taking array of indexes
# use 6 times second element of the array, to get a new array
print(new_mixed_array[[1, 1, 1, 1, 1, 1]])

# let's convert our array to a 3x3 two dimension matrix
two_dimensional_array = new_mixed_array.reshape(3, 3)
print(two_dimensional_array)

# let's pick the element
# same result = 6
print(two_dimensional_array[1][2])
print(two_dimensional_array[1, 2])
