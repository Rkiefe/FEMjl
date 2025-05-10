# A simple interpolation method to interpolate 1D data

module interp
export interp1

	function interp1(x::Vector{Float64},y::Vector{Float64},xq::Float64)
		i::Vector{Int32} = [0,0]

		found::Bool = false
		yq::Float64 = 0

		# Check if xq is in bounds of x
		if !(minimum(x) <= xq <= maximum(x))
			error("xq is not within x bounds")
		end

		# Check if xq is the last value of x
		if x[end] == xq
			yq = y[end]
			found = true

		elseif x[1] == xq
			yq = y[1]
			found = true

		else # Check all other values in array
			for j in 1:length(x)-1
				if x[j+1] > xq && x[j] < xq
					# found the two datapoints
					i[1] = j
					i[2] = j+1
					found = true
					break
				end
			end
		end

		if found && yq == 0
			# Make a line that connects x[i] and y[i]
			m::Float64 = (y[i[2]]-y[i[1]])/(x[i[2]]-x[i[1]])
			yq = m*(xq-x[i[1]]) + y[i[1]] 
		end

		return yq
	end # Interp1 over a value

	function interp1(x::Vector{Float64},y::Vector{Float64},xq::Vector{Float64})
		yq::Vector{Float64} = zeros(length(xq))
		@simd for i in 1:length(xq)
			yq[i] = interp1(x,y,xq[i])
		end

		return yq
	end # Interp1 over a vector

	# function interp2(x::Vector{Float64},y::Vector{Float64},z::Matrix{Float64},xq::Float64,yq::Float64)

		# 	i::Vector{Int32} = [0,0]
		# 	j::Vector{Int32} = [0,0]

		# 	iFound::Bool = false
		# 	jFound::Bool = false
		# 	zq::Float64 = 0

		# 	# Check if xq is in bounds of x
		# 	if !(minimum(x) <= xq <= maximum(x))
		# 		error("xq is not within x bounds")
		# 	end

		# 	# Check if yq is in bounds of x
		# 	if !(minimum(y) <= yq <= maximum(y))
		# 		error("yq is not within y bounds")
		# 	end

		# 	# Search for proper interval of x
		# 	if x[end] == xq # Check if xq is the last value of x
		# 		iFound = true

		# 	elseif x[1] == xq # Check if xq is the first value of x
		# 		iFound = true

		# 	else # Check all other values in array
		# 		for ind in 1:length(x)-1
		# 			if x[ind+1] > xq && x[ind] < xq
		# 				# found the two datapoints
		# 				i[1] = ind
		# 				i[2] = ind+1
		# 				iFound = true
		# 				break
		# 			end
		# 		end
		# 	end # Looking for x bounds

		# 	# Search for proper interval of y
		# 	if y[end] == yq # Check if xq is the last value of x
		# 		jFound = true

		# 	elseif y[1] == yq # Check if xq is the first value of x
		# 		jFound = true

		# 	else # Check all other values in array
		# 		for ind in 1:length(y)-1
		# 			if y[ind+1] > yq && y[ind] < yq
		# 				# found the two datapoints
		# 				j[1] = ind
		# 				j[2] = ind+1
		# 				jFound = true
		# 				break
		# 			end
		# 		end
		# 	end # Looking for x bounds

		# 	if iFound && jFound
		# 		# Do something...
				
		# 		# but for now
		# 		zq = 0.0
		# 	end

	# end # Interp2 over a value

end # End of Method 'interp'

