##############################################################################
############################ Pytorch 102 (20 pts) ############################
##############################################################################
"""
In this section, you will work on various PyTorch exercises 
that build upon basic tensor operations.

There are a total of 15 problems in this section.

Please make sure to check the input and return types 
as these will be graded automatically.
"""

import torch


##Problem 1. (1 pt)
def create_sample_tensor():
  """
  Return a torch Tensor of shape (2, 3) which is filled with zeros, except for
  element (0, 1) which is set to 10 and element (1, 0) which is set to 100.

  Inputs: None

  Returns:
  - Tensor of shape (2, 3) as described above.
  """
  x = None
  
  #############################################################################
  #       TODO: Implement this function. It should take one line.             #
  #############################################################################
  # Replace "pass" statement with your code
  x = torch.zeros((2, 3))
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x


##Problem 2. (1 pt)
def create_tensor_of_pi(M, N):
  """
  Returns a Tensor of shape (M, N) filled entirely with the value 3.14

  Inputs:
  - M, N: Positive integers giving the shape of Tensor to create

  Returns:
  - x: A tensor of shape (M, N) filled with the value 3.14
  """
  x = None
  #############################################################################
  #       TODO: Implement this function. It should take one line.             #
  #############################################################################
  
  x = torch.full((M, N), 3.14)
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x



##Problem 3. (1 pt)
def calculate_tensor_sum(x):
  """
  Calculate and return the sum of all elements in the input tensor.

  Inputs:
  - x: Tensor of any shape

  Returns:
  - result: Scalar tensor representing the sum of all elements in x.
  """
  result = None
  
  #################################YOUR CODE###################################
  result = x.sum()
  
  #############################################################################
  return result



##Problem 4. (1 pt)
def count_tensor_elements(x):
  """
  Count the number of scalar elements in a tensor x.

  For example, a tensor of shape (10,) has 10 elements.a tensor of shape (3, 4)
  has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

  You may not use the functions torch.numel(x) or x.numel(). The input tensor should
  not be modified.

  Inputs:
  - x: A tensor of any shape

  Returns:
  - num_elements: An integer giving the number of scalar elements in x
  """
  num_elements = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #   You CANNOT use the built-in functions torch.numel(x) or x.numel().      #
  #############################################################################
  n, m = x.shape
  num_elements = n * m
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return num_elements



##Problem 5. (2 pt)
def divisible_by_five(start, end):
  """
  Return a 1D tensor containing numbers from 'start' to 'end' (inclusive) that are 
  divisible by five. The result should be of dtype torch.float32.

  Inputs:
  - start: Integer, starting value
  - end: Integer, end value

  Returns:
  - result: 1D tensor containing values divisible by 5 between start and end, inclusive.
  """
  assert start <= end
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  result = torch.tensor([i for i in range(start, end + 1) if i % 5 == 0], dtype=torch.float32)
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 6. (1 pt)
def slice_tensor_by_steps(x):
  """
  Given a 2D tensor x, extract and return specific elements using step-based 
  slicing. You should return:
  - Even-indexed rows (0, 2, 4, ...) 
  - Odd-indexed columns (1, 3, 5, ...)

  Inputs:
  - x: Tensor of shape (M, N) with M >= 4, N >= 4.

  Returns:
  - result: A 2D tensor containing the elements in the even-indexed rows 
    and odd-indexed columns.
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  result = x[::2, 1::2]
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 7. (1 pt)
def assign_to_slice(x):
  """
  Given a tensor x of shape (5, 5), mutate its center 3x3 block to be filled with the value 9.

  Input:
  - x: Tensor of shape (5, 5)

  Returns:
  - x: Mutated tensor with center 3x3 block filled with 9.
  """
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  x[1:4, 1:4] = 9
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return x



##Problem 8. (1 pt)
def transpose_tensor(x):
  """
  Return the transpose of a 2D tensor.

  Input:
  - x: Tensor of shape (M, N)

  Returns:
  - result: Transposed tensor of shape (N, M)
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  result = x.t()
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 9. (1 pt)
def extract_main_diagonal(x):
  """
  Extract the main diagonal of a 2D square tensor.

  Inputs:
  - x: Tensor of shape (N, N)

  Returns:
  - result: 1D tensor representing the main diagonal of x.
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  result = x.diagonal()
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 10. (1 pt)
def count_nonzero_elements(x):
  """
  Return the number of nonzero elements in the input tensor.

  Inputs:
  - x: A tensor of any shape

  Returns:
  - result: Integer representing the number of nonzero elements in x.
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  middle = (x != 0).sum()
  result = middle.item()  # Convert tensor to Python integer
#   print(f"{middle} non-zero elements found in the tensor.")
#   print(f"Total non-zero elements: {result}")
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 11. (1 pt)
def create_identity_matrix(n):
  """
  Create an identity matrix of shape (n, n).

  Inputs:
  - n: Integer, size of the identity matrix

  Returns:
  - result: Identity matrix of shape (n, n)
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  result = torch.eye(n)
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 12. (2 pts)
def reshape_practice_v2(x):
    """
    Given an input tensor of shape (36,), return a reshaped tensor y of shape
    (4, 9) such that

    y = [
      [x[0], x[1], x[2],  x[12],  x[13], x[14], x[24], x[25], x[26]],
      [x[3], x[4], x[5],  x[15], x[16], x[17], x[27], x[28], x[29]],
      [x[6], x[7], x[8],  x[18], x[19], x[20], x[30], x[31], x[32]],
      [x[9], x[10], x[11], x[21], x[22], x[23], x[33], x[34], x[35]],
    ]

    You must construct y by performing a sequence of reshaping operations on x
    (view, t, transpose, permute, contiguous, reshape, etc). The input tensor
    should not be modified.

    Input:
    - x: A tensor of shape (36,)

    Returns:
    - y: A reshaped version of x of shape (4, 9) as described above.
    """
    y = None
    #############################################################################
    #                    TODO: Implement this function                          #
    #############################################################################
    # Replace "pass" statement with your code
    y = x.view(3,4,3)
    y = y.permute(1, 0, 2).reshape(4, 9)
    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    return y



##Problem 13. (2 pt)
def zero_min_in_each_row(x):
  """
  Return a copy of x where the minimum value in each row has been set to zero.

  Inputs:
  - x: A 2D tensor of shape (M, N)

  Returns:
  - result: A copy of x with the minimum value in each row replaced by 0.
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  M, N = x.shape
  result = x.clone()  # Create a copy of x
  for i in range(M):
    min_value = result[i].min()
    min_index = (result[i] == min_value).nonzero(as_tuple=True)[0]
    result[i, min_index] = 0  # Set the minimum value to zero
  
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



##Problem 14. (2 pts)
def batch_matrix_multiply(x, y, use_loop=True):
  """
  Perform batched matrix multiplication between the tensors x and y. 
  If use_loop=True, then you should use an explicit loop over the batch
  dimension B. If loop=False, then you should instead compute the batched
  matrix multiply without an explicit loop using a single PyTorch operator.

  Inputs:
  - x: Tensor of shape (B, N, M)
  - y: Tensor of shape (B, M, P)
  - use_loop: Boolean indicating whether to use a loop.

  Returns:
  - result: Tensor of shape (B, N, P) resulting from batched matrix multiplication.
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  # Replace "pass" statement with your code
  if use_loop:
    B, N, M = x.shape
    _, M2, P = y.shape
    assert M == M2, "Inner dimensions must match for matrix multiplication."
    result = torch.zeros((B, N, P), dtype=x.dtype)
    for i in range(B):
      result[i] = x[i].mm(y[i])
  else:
    result = torch.bmm(x, y)  # Batched matrix multiplication without loop
      
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result



def mm_on_cpu(x, w):
  """
  (helper function) Perform matrix multiplication on CPU.
  PLEASE DO NOT EDIT THIS FUNCTION CALL.

  Input:
  - x: Tensor of shape (A, B), on CPU
  - w: Tensor of shape (B, C), on CPU

  Returns:
  - y: Tensor of shape (A, C) as described above. It should not be in GPU.
  """
  y = x.mm(w)
  return y



##Problem 15. (2 pt)
def gpu_matrix_multiplication(x, w):
  """
  Perform matrix multiplication on GPU.

  Specifically, you should (i) place each input on GPU first, and then
  (ii) perform the matrix multiplication operation. Finally, (iii) return the
  final result which is on CPU.

  Inputs:
  - x: Tensor of shape (A, B), on CPU
  - w: Tensor of shape (B, C), on CPU

  Returns:
  - result: Result of matrix multiplication, moved back to CPU.
  """
  result = None
  #############################################################################
  #                    TODO: Implement this function                          #
  #############################################################################
  x_gpu = x.to("cuda")       # ① CPU → GPU
  w_gpu = w.to("cuda")       # ② CPU → GPU
  result_gpu = x_gpu.mm(w_gpu)  # ③ GPU 상에서 곱하기
  result = result_gpu.to("cpu")  # ④ GPU → CPU
  #############################################################################
  #                            END OF YOUR CODE                               #
  #############################################################################
  return result