require 'optim'
require 'nn'

function analyze_model(model, criterion, grid, data) 
	
	local param,gradParam=model:getParameters()
	gradParam:zero()
	local output=model:forward(data[1])
	local loss=criterion:forward(output,data[2])
	local dL_do=criterion:backward(output,data[2])
	model:backward(data[1],dL_do)
	
	


end
