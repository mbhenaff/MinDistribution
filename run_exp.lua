require 'optim'
require 'nn'
require 'image'
require 'finite_diff'

printf = function(s,...)
	return io.write(s:format(...))
end

function eigIndex(eigenvals)
   local index = 0
   for i = 1,eigenvals:size(1) do
      if eigenvals[i] > 0 then
	 index = index + 1
      elseif eigenvals[i] < 0 then
	 index = index - 1
      end
   end
   return index
end


function analyze_model(model, criterion, nPtsPerDim, gridRange, data, gradThreshold) 
	local criticalPts={}
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
		
		local function feval(x)
			if x ~= param then
				param:copy(x)
			end
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
			return loss, gradParam
		end
		loss,_ = feval(param)
		lossValues[indx:storage()]=loss
		local gradNorm=gradParam:norm()
		gradValues[indx:storage()]=gradNorm
		if gradNorm < gradThreshold then
			hessian=hessianPerturbationNet(feval, param, 1e-3)
			e=torch.eig(hessian)[{{},1}]
			criticalPts[#criticalPts+1]={eigenvals=e,param=param:clone(),indx=indx:clone()}
		end
	end
	return lossValues,gradValues,criticalPts
end

nSamples=100
dimInputs=1
model = nn.Sequential()
model:add(nn.Mul(dimInputs))
model:add(nn.Mul(dimInputs))
criterion=nn.MSECriterion()
data=torch.randn(nSamples):repeatTensor(2,1)
data=data:reshape(data:size(1),data:size(2),1)

loss,gradNorm,criticalPts=analyze_model(model,criterion,50,4,data,0.3)
gradNorm:reshape(gradNorm:nElement())
gnuplot.hist(gradNorm,100)

imdisp = torch.Tensor(3,loss:size(1), loss:size(2))
for i = 1,3 do
   imdisp[i]:copy(loss)
end

local maxLossDisplay = 3
for i = 1,#criticalPts do
   local indx = criticalPts[i]['indx']
   local eigenvals = criticalPts[i]['eigenvals']
   eigidx = eigIndex(eigenvals)
   eigidx_normalized = (eigidx + eigenvals:size(1))/(eigenvals:size(1)*2)
   imdisp[1][indx:storage()] = maxLossDisplay*eigidx_normalized
   imdisp[2][indx:storage()] = maxLossDisplay*(1-eigidx_normalized)
   imdisp[3][indx:storage()] = 0
end
image.display{image=imdisp, zoom=5, max=maxLossDisplay}
