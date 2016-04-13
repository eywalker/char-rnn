local RNN = {}

-- creates a multi-layered RNN with n layers
-- the network works on the input of size input_size
-- and the n hidden states from the previous time step
-- each layer produces output of size rnn_size
function RNN.rnn(input_size, rnn_size, n, dropout)
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}

  for L = 1,n do
    -- hidden state of layer L is at index L+1
    local prev_h = inputs[L+1]
    if L == 1 then
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else
      x = outputs[(L-1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return RNN
