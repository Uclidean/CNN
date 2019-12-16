import numpy as np
def conv_forward_naive(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input. 
            

        During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
        along the height and width axes of the input. Be careful not to modfiy the original
        input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
        H' = (W - HH )
        - cache: (x, w, b, conv_param)
        """
        
        pad = conv_param['pad'] # Padding number
        stride = conv_param['stride'] # Stride number
        input_N, input_c, input_h, input_w = x.shape # Input features
        x_padded = np.pad(x, ((0,0),(0,0),(pad,pad), # Input with padding applied to it
                        (pad,pad)), 'constant')
        N, i_c, i_h, i_w = x_padded.shape # Padded input features 
        F, f_c, f_h, f_w = w.shape # Filter features
        a_h = 1 + (input_h + 2*pad - f_h) // stride # Height of depth slice of activation volume
        a_w = 1 + (input_w + 2 * pad - f_w) // stride # Width of depth slice of activation volume
        activation_map = np.zeros((N,F,a_h,a_w)) # Activation volume for each input

        for j in range(N): # Iterate through all inputs
            curr_y = out_y = 0 # Keeps track of position of filter vertically
            while (curr_y + f_h <= i_h): # Slide filter through input vertically
                curr_x = out_x = 0 # Keeps track of position of filter horizontally
                while (curr_x + f_w <= i_w): # Slide filter through input horizontally
                  y = x_padded[j,::,curr_y:curr_y+4,curr_x:curr_x+4]*w # Convolution operation
                  for i in range(F): # Iterate through all filters
                      activation_map[j, i, out_y, out_x] = np.sum(y[i])+b[i] # Sum the result of each convolution operation with relevant bias term
                  curr_x += stride 
                  out_x += 1
                curr_y += stride
                out_y += 1
        cache = (x, w, b, conv_param)
        return activation_map, cache

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    pool_height = pool_param['pool_height'] # Height of pool
    pool_width = pool_param['pool_width'] # Width of pool
    stride = pool_param['stride'] # Stride number
    N, C, H, W = x.shape # Input features
    HH = 1 + (H - pool_height) // stride # Height of output
    WW = 1 + (W - pool_width) // stride # Width of output
    out = np.zeros((N, C, HH, WW))
    for n in range(N): # Iterate through all inputs
      out_y = curr_y = 0
      while (curr_y + pool_height <= H): # Slide pool vertically
        out_x = curr_x = 0
        while (curr_x + pool_width <= W): # Slide pool horizontally
          out[n, :, out_y, out_x] =  np.max(x[n, :, curr_y:curr_y + pool_height,
                                               curr_x:curr_x + pool_width], axis = (1,2))
          out_x +=1
          curr_x += stride
        out_y +=1
        curr_y += stride

    cache = (x, pool_param)
    return out, cache
