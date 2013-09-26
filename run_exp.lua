require 'optim'
require 'nn'

function loopOnGrid(nValPerDim, v, i)
   if v[i] < nValPerDim then
      v[i] = v[i] + 1
      if i < v:size(1) then
	 return i+1
      else
	 return i
      end
   else
      v[i] = 0
      if i > 0 then
	 return loopOnGrid(nValPerDim, v, i+1)
      else
	 return nil
      end
   end	 
end

--debug
v = torch.zeros(3)
i = 1
print(v)
while i ~= nil do
   loopOnGrid(2, v, i)
   print (v)
end

function analyze_model(model, criterion, grid, data) 
	
	local param,gradParam=model:getParameters()
	gradParam:zero()
	local output=model:forward(data[1])
	local loss=criterion:forward(output,data[2])
	local dL_do=criterion:backward(output,data[2])
	model:backward(data[1],dL_do)
	
	


end
