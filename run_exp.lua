require 'optim'
require 'nn'
require 'image'


printf = function(s,...)
	return io.write(s:format(...))
end

function analyze_model(model, criterion, nPtsPerDim, gridRange, data) 
	local grid = torch.linspace(-gridRange/2,gridRange/2,nPtsPerDim)
	local param,gradParam=model:getParameters()
	local D=param:size(1)
	local lossValues=torch.Tensor((torch.ones(D)*nPtsPerDim):long():storage())
	local gradValues=torch.Tensor((torch.ones(D)*nPtsPerDim):long():storage())
	local indx = torch.LongTensor(D)

	for i=0,nPtsPerDim^D-1 do
		xlua.progress(i,nPtsPerDim^D-1)
        	for j=1,D do
			indx[j] = math.mod(math.floor(i/nPtsPerDim^(j-1)),nPtsPerDim)+1
                	param[j] = grid[ indx[j] ]
		end
		gradParam:zero()

		local loss=0
		for k=1,data[1]:size(1) do
			local input=data[1][k]
			local target=data[2][k]
			local output=model:forward(input)
			loss=loss+criterion:forward(output,target)
			local dL_do=criterion:backward(output,target)
			model:backward(input,dL_do)
		end
		loss = loss/data[1]:size(1)
		gradParam:div(data[1]:size(1))
		lossValues[indx:storage()]=loss
		gradValues[indx:storage()]=gradParam:norm()
	end
	return lossValues,gradValues
end

nSamples=100
dimInputs=1
model = nn.Sequential()
model:add(nn.Mul(dimInputs))
model:add(nn.Mul(dimInputs))
criterion=nn.MSECriterion()
data=torch.randn(nSamples):repeatTensor(2,1)
data=data:reshape(data:size(1),data:size(2),1)

loss,gradNorm=analyze_model(model,criterion,50,4,data)
gradNorm:reshape(gradNorm:nElement())
image.display(loss)
gnuplot.hist(gradNorm,100)



