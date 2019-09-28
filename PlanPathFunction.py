import numpy as np

def plan_path(start, end):
    number_of_steps_curve = 30
    number_of_steps_drop = 20
    lowering_height = 0.3
    
    path = np.zeros((number_of_steps_curve + number_of_steps_drop,3)) # initialise array of zeros
    
    step_xyz = [0,0,0] #initialise step

    for i in range (0,3): 
        path[0, i] = start[i] #set first step as start position
        step_xyz[i] = ((start[i] + end[i])/float(number_of_steps_curve)) #define step distance
        
    step_xyz[2] = (((start[2]+ (end[2]+lowering_height)))/number_of_steps_curve) #define z step distance
    
    for i in range(1,(number_of_steps_curve + number_of_steps_drop)): #incrememnt x and y path according to step_xyz
        if (i <= number_of_steps_curve):
            path[i,0] = path[i-1,0]+step_xyz[0]
            path[i,1] = path[i-1,1]+step_xyz[1] 
            path[i,2] = ((i**(1./3)) / ((number_of_steps_curve-1)**(-(2./3)))) * (step_xyz[2]) #incremment z path according to curve equation
        else:                                                           #keep x and y the same
            path[i,0] = path[i-1,0]                     
            path[i,1] = path[i-1,1]
            path[i,2] = path[i-1,2] - (lowering_height / number_of_steps_drop)      #lower z according to drop height
    path[number_of_steps_curve + number_of_steps_drop-1] = end                      #ensure final value is equal to target (without this for low values of number_steps_drop the final path can have a small z error)
    return path

test = plan_path((0.75, 0, 0), (-0.6, -1, -1))
print(test)