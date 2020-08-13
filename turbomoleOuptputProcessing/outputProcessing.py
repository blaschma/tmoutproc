
import numpy as np
import math
from scipy.sparse import coo_matrix
import scipy.sparse
import re as r
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.linalg import eig
from functools import partial
from multiprocessing import Pool




def outer_product(vector1, vector2):
	#print("length " + str(len(vector1)))	
	#print(np.outer(vector1,vector2))
	return np.outer(vector1,vector2)

#return string of float f with giben precision prec and number of digits in exponent exp_digits
def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%sE%+0*d"%(mantissa, exp_digits+1, int(exp))

# gives a measure of asymmetry by calculatin max(|matrix-matrix^T|) (absvalue elementwise)
def measure_asymmetry(matrix):
	return np.max(np.abs(matrix-np.transpose(matrix)))

#gives a measure of difference of two matrices 
def measure_difference(matrix1, matrix2):
	min_error = np.min(diff)
	print("min "  + str(min))
	max_error = np.max(diff)
	print("max "  + str(max))
	avg_error = np.mean(diff)
	print("avg "  + str(avg))
	var_error = np.sqrt(np.std(diff))
	print("var "  + str(var))
	return min_error,max_error,avg_error,var_error




#calculates the A matrix using mos file, reads eigenvalues from qpenergies (->DFT eigenvalues), input: mosfile and prefixlength (number of lines in the beginning), eigenvalue source (qpenergiesKS or qpenergiesGW), 
#if eigenvalue_path is specified you can give custom path to qpenergies and col to read
#returns matrix in scipy.sparse.csc_matrix and eigenvalue list
def calculate_A(filename,prefix_length=2, eigenvalue_source = "mos", eigenvalue_path = "", eigenvalue_col = 2):
	#length of each number
	n = 20
	level = 0
	C_vec = list()
	A_i = 0
	counter = 0
	eigenvalue_old = -1
	eigenvalue = 0
	beginning = True
	eigenvalue_list= list()
	#if eigenvalue source = qpEnergies
	if(eigenvalue_source == "qpenergiesKS" and eigenvalue_path == ""):
		eigenvalue_list = read_qpenergies("qpenergies.dat", col = 1)
	elif(eigenvalue_source == "qpenergiesGW" and eigenvalue_path == ""):
		eigenvalue_list = read_qpenergies("qpenergies.dat", col = 2)
	elif(eigenvalue_path != ""):
		print("custom path ")
		eigenvalue_list = read_qpenergies(eigenvalue_path, col = eigenvalue_col)
		#print(eigenvalue_list)


	#open mos file and read linewise
	with open(filename) as f:
		for line in f:
			#skip prefix lines
			#find level and calculate A(i)
			if(counter>prefix_length):
				index1 = (line.find("eigenvalue="))
				index2 = (line.find("nsaos="))
				#eigenvalue and nsaos found -> new orbital
				if(index1 != -1 and index2 != -1):
					level += 1 
					
					#find eigenvalue of new orbital and store eigenvalue of current orbital in eigenvalue_old
					if((beginning == True and eigenvalue_source == "mos") and eigenvalue_path == ""):
						eigenvalue_old = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))						
						beginning = False
					elif(eigenvalue_source == "mos" and eigenvalue_path == ""):
						#print("eigenvalue from mos")
						eigenvalue_old = eigenvalue
						eigenvalue_list.append(eigenvalue_old)
						#if(level != 1128):
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
					#if eigenvalues are taken from qpenergies
					elif(eigenvalue_source == "qpenergiesKS" or eigenvalue_source == "qpenergiesGW" or eigenvalue_path != ""):
						if(beginning == True):
							beginning = False
							pass
						else:
							#print("eigenvalue from eigenvaluelist")
							eigenvalue_old = eigenvalue_list[level-2]

												 
					#find nsaos of new orbital
					nsaos = (int(line[(index2+len("nsaos=")):len(line)]))

					#create empty A_i matrix
					if(isinstance(A_i,int)):
						A_i = np.zeros((nsaos,nsaos))						
						

					#calculate A matrix by adding A_i -> processing of previous orbital		
					if(len(C_vec)>0):			
						print("take " + str(eigenvalue_old))						
						A_i += np.multiply(outer_product(C_vec,C_vec),eigenvalue_old)						
						C_vec = list()					
					#everything finished (beginning of new orbital)
					continue
				
				
				line_split = [line[i:i+n] for i in range(0, len(line), n)]							
				C_vec.extend([float(line_split[j].replace("D","E"))  for j in range(0,len(line_split)-1)][0:4])
				#print(len(C_vec))
				
			counter += 1
			#for testing
			if(counter > 300):
				#break
				pass
	#handle last mos
	if(eigenvalue_source == "qpenergiesKS" or eigenvalue_source == "qpenergiesGW"):
		eigenvalue_old = eigenvalue_list[level-1]
		print("lasteigen " + str(eigenvalue_old))
	
	A_i += np.multiply(outer_product(C_vec,C_vec),eigenvalue_old)

	A_i = scipy.sparse.csc_matrix(A_i, dtype = float)
	print("A_mat symmetric: " + str(np.allclose(A_i.toarray(), A_i.toarray(), 1e-04, 1e-04)))
	print("A_mat asymmetry measure " + str(measure_asymmetry(A_i.toarray())))
	print("------------------------")	
	return A_i,eigenvalue_list

#read packed symmetric matrix from file and creates matrix
#returns matrix in scipy.sparse.csc_matrix
def read_packed_matrix(filename):	
	#matrix data
	data = []
	#matrix rows, cols
	i = []
	j = []

	counter =-1
	col = 0
	col_old = 0
	row = 0;
	#open smat_ao.dat and create sparse matrix
	with open(filename) as f:
		for line in f:
			#skip first line
			if(counter == -1):
				counter+=1
				continue
			#line_split = [i.split() for i in line]
			line_split = r.split('\s+', line)
			#print(counter)
			#print(line_split[2])

			matrix_entry = np.float(line_split[2])
			#print(eformat(matrix_entry,100,5))
			#print(line_split[2])
			#print(eformat(np.longdouble(line_split[2]),100,5))
			#print("-----")
			#matrix_entry = round(matrix_entry,25)
			matrix_enty = round(float(line_split[2]),24)

			

			#calculate row and col
			
			if(col == col_old+1):
				col_old = col
				col = 0;
				row += 1
			#print("setting up row " + str(counter))
			#print(row,col)
			#skip zero elemts
			if(matrix_entry != 0.0):
				data.append(matrix_entry)
				#print(matrix_entry)
				#print("------")
				i.append(col)
				j.append(row)
				#symmetrize matrix
				if(col != row):
					data.append(matrix_entry)
					i.append(row)
					j.append(col)
					pass
			col += 1	

			
			counter+=1
			#for testing
			if(counter>25):
				#break
				pass
	
	coo = coo_matrix((data, (i, j)), shape=(row+1, row+1))
	csc = scipy.sparse.csc_matrix(coo, dtype = float)	
	print("S_mat symmetric: " + str(np.allclose(csc.toarray(), csc.toarray(), 1e-04, 1e-04)))
	print("S_mat asymmetry measure " + str(measure_asymmetry(csc.toarray())))
	print("------------------------")
	return(csc)



#write symmetric scipy.sparse.csc_matrix in  in packed storage form
def write_matrix_packed(matrix, filename="test"):
	print("writing packed matrix")
	num_rows = matrix.shape[0]
	num_elements_to_write = (num_rows**2+num_rows)/2
	

	col = 0
	row = 0
	element_counter = 1
	f = open(filename, "w")

	line_beginning_spaces = ' ' * (12 - len(str(num_elements_to_write)))
	f.write(line_beginning_spaces + str(num_elements_to_write) + "      nmat\n")

	for r in range(0,num_rows):
		#print("writing row " +str(r))
		num_cols = r+1		
		for c in range(0, num_cols):
			matrix_element = matrix[r,c]
			line_beginning_spaces = ' ' * (20 -len(str(int(element_counter))))
			if(matrix_element>=0):
				line_middle_spaces = ' ' * 16
			else:
				line_middle_spaces = ' ' * 15
			
			f.write(line_beginning_spaces + str(int(element_counter)) + line_middle_spaces + eformat(matrix_element, 14,5) + "\n")			
			element_counter +=1
	f.close()
	print("writing done")

#read qpenergies.dat and returns data of given col as list, skip_lines in beginning of file, convert to eV
#col = 0 qpenergies, col = 1 Kohn-Sham, 2 = GW
def read_qpenergies(filename, col=1, skip_lines=1):

	har_to_ev = 27.21138602
	qpenergies = list()	
	datContent = [i.strip().split() for i in open(filename).readlines()[(skip_lines-1):]]
	#print(datContent)
	#datContent = np.transpose(datContent)
	#print(datContent)	

	for i in range(skip_lines, len(datContent)):
		energy = float(datContent[i][col])/har_to_ev
		#print("qpenergy " + str(energy))
		qpenergies.append(energy)
		pass	
	return qpenergies





#write mos file, requires python3
def write_mos_file(eigenvalues, eigenvectors, filename="mos_new.dat"):
	f = open(filename, "w")
	#header
	f.write("$scfmo    scfconv=7   format(4d20.14)\n")
	f.write("# SCF total energy is    -6459.0496515472 a.u.\n") 
	f.write("#\n")   
	for i in range(0, len(eigenvalues)):
		print("eigenvalue " + str(eigenvalues[i]) + "\n")
		first_string = ' ' * (6-len(str(i))) + str(i+1) + "  a      eigenvalue=" + eformat(eigenvalues[i], 14,2) + "   nsaos=" + str(len(eigenvalues))
		f.write(first_string + "\n")
		j = 0
		while j<len(eigenvalues):
			for m in range(0,4):
				num = eigenvectors[m+j,i]
				#string_to_write = f"{num:+20.13E}".replace("E", "D")
				f.write(string_to_write)
				#f.write(eformat(eigenvectors[m+j,i], 14,2).replace("E", "D"))
			f.write("\n")
			j = j +4
			#print("j " + str(j))
	f.write("$end")
	f.close()

def read_mos_file(filename, skip_lines=1):
	#length of each number
	n = 20
	level = 0
	C_vec = list()	
	counter = 0
	eigenvalue_old = -1
	eigenvalue = 0
	beginning = True
	eigenvalue_list= list()	
	eigenvector_list= -1	


	#open mos file and read linewise
	with open(filename) as f:
		for line in f:
			#skip prefix lines
			#find level and calculate A(i)
			if(counter>skip_lines):
				index1 = (line.find("eigenvalue="))
				index2 = (line.find("nsaos="))
				#eigenvalue and nsaos found -> new orbital
				if(index1 != -1 and index2 != -1):
					level += 1 
					#find nsaos of new orbital
					nsaos = (int(line[(index2+len("nsaos=")):len(line)]))	

					#find eigenvalue of new orbital and store eigenvalue of current orbital in eigenvalue_old
					if(beginning == True):
						eigenvalue_old = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
						eigenvector_list = np.zeros((nsaos,nsaos),dtype=float)						
						beginning = False
					else:
						eigenvalue_old = eigenvalue	
						
						eigenvalue = float(line[(index1+len("eigenvalue=")):index2].replace("D","E"))
												 
													
						

					#calculate A matrix by adding A_i -> processing of previous orbital		
					if(len(C_vec)>0):							
						eigenvalue_list.append(eigenvalue_old)	
						#print("level " + str(level))
						eigenvector_list[:,(level-2)] = C_vec					
						C_vec = list()					
					#everything finished (beginning of new orbital)
					continue
				
				
				line_split = [line[i:i+n] for i in range(0, len(line), n)]							
				C_vec.extend([float(line_split[j].replace("D","E"))  for j in range(0,len(line_split)-1)][0:4])
				#print(len(C_vec))
				
			counter += 1
			
	#handle last mos
	eigenvalue_list.append(eigenvalue_old)	
	eigenvector_list[:,(nsaos-1)] = C_vec
	
	return eigenvalue_list,eigenvector_list


#helper function for trace_mo
def scalarproduct(index_to_check, para):	
	ref_mos = para[0]
	input_mos = para[1]
	#abweichung von urspruenglicher Position
	tolerance = para[2]
	s_mat = para[3]
	most_promising = -1
	#print(ref_mos)	
	candidate_list = list()
	index_list = list()
	#s_mat = read_packed_matrix("./data/smat_ao.dat")
	for i in range(0, ref_mos.shape[1]):		
		#print("scalar " + str(float(scalar_product)))		
		if(tolerance != -1  and (i < index_to_check+tolerance) and (i > index_to_check-tolerance)):
			#print("treffer")
			#print("ref mos shape " + str(ref_mos[:,i].shape))
			#print("smat shape oben " + str(s_mat.shape))

			scalar_product = list(np.dot(s_mat, ref_mos[:,i]).flat)
			#print("scalar produce shapeo " + str(scalar_product.shape))
			scalar_product = np.dot(np.transpose(input_mos[:,index_to_check]), scalar_product)
			#print("scalar " + str(float(scalar_product)))	
			if(np.isclose(float(scalar_product),1.0,atol=0.4)):
				candidate_list.append(scalar_product)
				index_list.append(i)
		elif(tolerance == -1):
			print("ref mos shape " + str(ref_mos[:,i].shape))
			print("smat shape oben " + str(s_mat.shape))

			scalar_product = list(np.dot(s_mat, ref_mos[:,i]).flat)
			#print("scalar produce shapeo " + str(scalar_product.shape))
			scalar_product = np.dot(np.transpose(input_mos[:,index_to_check]), scalar_product)
			#print("scalar " + str(float(scalar_product)))	
			if(np.isclose(float(scalar_product),1.0,atol=0.4)):
				candidate_list.append(scalar_product)
				index_list.append(i)
		#if they cannot be traced
		candidate_list.append(-1)
		index_list.append(-1)

	most_promising = [x for _,x in sorted(zip(candidate_list,index_list))]
	most_promising = most_promising[-1]
	#print("most_promising " + str(most_promising))	
	return most_promising
	

#traces mos from input_mos with reference to ref_mos (eg when order has changed) 
#calculates the scalarproduct of input mos with ref_mos and takes the highest match (close to 1)
#ref_mos, input_mos : eigenvectors, tolerance maximum difference in mo index			
def trace_mo(ref_mos, input_mos, tolerance=-1):	
	input_mos_list = list()
	#prepare
	for i in range(0, ref_mos.shape[0]):
		input_mos_list.append(input_mos[:,i])
	
	print("filling pool")
	p = Pool(16)	
	s_mat = read_packed_matrix("./data/smat_ao.dat").todense()
	para = (ref_mos, input_mos, tolerance, s_mat)

	#all eigenvectors

	index_to_check = range(0, input_mos.shape[0])
	result = p.map(partial(scalarproduct, para=para), index_to_check)		
	print("done")
	return result





#diagonalizes f mat, other eigenvalues can be used (eigenvalue_list)
def diag_F(path, eigenvalue_list = list()):

	print("read f")
	F_mat_file = read_packed_matrix(path)
	print(F_mat_file.shape)

	print("diag F")	
	
	#def foo(matrix):
	#	np.linalg.eigh(matrix)
	#t = timeit.Timer(functools.partial(foo, F_mat_file.todense()))
	#print("Timing")  
	#print t.timeit(number = 1)
	

	eigenvalues, eigenvectors = np.linalg.eigh(F_mat_file.todense())
	print("diag F done")

	print("calc fmat ")
	#take eigenvalues from diagonalization or external eigenvalues (e.g from qpenergies)
	if(len(eigenvalue_list) == 0):
		eigenvalues = np.diag(eigenvalues)
	else:		
		eigenvalues = np.diag(eigenvalue_list)


	a_mat = eigenvalues * np.transpose(eigenvectors)
	f_mat = eigenvectors * a_mat
	print("calc fmat done")

	#force H_LR = 0
	'''
	for i in range(0, len(eigenvalues)):
		for j in range(0, len(eigenvalues)):
			if(i > 729 and i < 1128 and j > 129 and j < 528):
				f_mat[i,j] = 0
				f_mat[j,i] = 0
	'''
	return f_mat
	



