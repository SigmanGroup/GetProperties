#code written by Brittany C. Haas and Melissa A. Hardy (adapted from David B. Vogt's get_properties_pandas.py, adapted from Tobias Gensch)

import re
import math
import itertools
import multiprocessing

import dbstep.Dbstep as db

from utils import get_filecont

#import matplotlib.pyplot as plt
#from matplotlib import rcParams


zero_pattern = re.compile("zero-point Energies")
cputime_pattern = re.compile("Job cpu time:")
walltime_pattern = re.compile("Elapsed time:")
volume_pattern = re.compile("Molar volume =")


frqs_pattern = re.compile("Red. masses")
frqsend_pattern = re.compile("Thermochemistry")


def get_sterimol_dbstep(dataframe, sterimol_list): #uses DBSTEP to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']

            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = []
            for sterimol in sterimol_list:
                atomnum_list = [] #the atom numbers use to collect sterimol values (i.e. [18 16 17 15]) are collected from the df using the input list (i.e. [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]

            #checks for if the wrong number of atoms are input or input is not of the correct form
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "****Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for Sterimol"
                if error != "": print(error)

            #this collects Sterimol values for each pair of inputs
            sterimol_out = []
            fp = log_file + str(".log")
            for sterimol in sterimolnums_list:
                sterimol_values = db.dbstep(fp,atom1=int(sterimol[0]),atom2=int(sterimol[1]),commandline=True,verbose=False,sterimol=True,measure='grid')
                sterimol_out.append(sterimol_values)

            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)

            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Sterimol_B1_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].Bmin,
                         'Sterimol_B5_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].Bmax,
                         'Sterimol_L_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].L}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire DSBTEP Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data",
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data",
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("DBSTEP Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_sterimol2vec(dataframe, sterimol_list, end_r, step_size): #uses DBSTEP to calculate sterimol Bmin and Bmax for two input atoms at intervals from 0 to end_r at step_size
    sterimol_dataframe = pd.DataFrame(columns=[])
    num_steps = int((end_r)/step_size + 1)
    radii_list = [0 + step_size*i for i in range(num_steps)]

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']

            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = []
            for sterimol in sterimol_list:
                atomnum_list = [] #the atom numbers use to collect sterimol values (i.e. [18 16 17 15]) are collected from the df using the input list (i.e. [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]

            #checks for if the wrong number of atoms are input or input is not of the correct form
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                if error != "": print(error)

            #this collects Sterimol values for each pair of inputs
            sterimol2vec_out = []
            fp = log_file + str(".log")
            for sterimol in sterimolnums_list:
                sterimol2vec_values = db.dbstep(fp,atom1=int(sterimol[0]),atom2=int(sterimol[1]),scan='0.0:{}:{}'.format(end_r,step_size),commandline=True,verbose=False,sterimol=True,measure='grid')
                sterimol2vec_out.append(sterimol2vec_values)

            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)

            scans = radii_list
            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                for i in range(0, len(scans)):
                    entry = {'Sterimol_Bmin_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": sterimol2vec_out[a].Bmin[i],
                             'Sterimol_Bmax_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": sterimol2vec_out[a].Bmax[i]}
                    row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire DSBTEP Sterimol2Vec parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    for i in range(0, len(scans)):
                        entry = {'Sterimol_Bmin_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": "no data",
                                'Sterimol_Bmax_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": "no data"}
                        row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("DBSTEP Sterimol2Vec function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_enthalpies(dataframe): # gets thermochemical data from freq jobs
    enthalpy_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont = get_filecont(log_file) #read the contents of the log file

            evals = []
            error = "no thermochemical data found;;"
            e_hf,ezpe,h,g = 0,0,0,0
            for i in range(len(filecont)-1): #uses the zero_pattern that denotes this section to gather relevant energy terms
                if zero_pattern.search(filecont[i]):
                    e_hf = round(-eval(str.split(filecont[i-4])[-2]) + ezpe,6)
                    evals.append(e_hf)
                    ezpe = eval(str.split(filecont[i])[-1])
                    evals.append(ezpe)
                    h = eval(str.split(filecont[i+2])[-1])
                    evals.append(h)
                    g = eval(str.split(filecont[i+3])[-1])
                    evals.append(g)
                    error = ""

            #this adds the data from the energy_values list (evals) into the new property df
            row_i = {'ZP_correction(Hartree)': evals[0], 'E_ZPE(Hartree)': evals[1], 'H(Hartree)': evals[2], 'G(Hartree)': evals[3]}
            #print(row_i)

            enthalpy_dataframe = enthalpy_dataframe.append(row_i, ignore_index=True)
        except:
            print('Unable to acquire enthalpies for:', row['log_name'], ".log")
    print("Enthalpies function has completed")
    return(pd.concat([dataframe, enthalpy_dataframe], axis = 1))

class IR:
    def __init__(self,filecont,start,col,len):
        self.freqno = int(filecont[start].split()[-3+col])
        self.freq = float(filecont[start+2].split()[-3+col])
        self.int = float(filecont[start+5].split()[-3+col])
        self.deltas = []
        atomnos = []
        for a in range(len-7):
            atomnos.append(filecont[start+7+a].split()[1])
            x = float(filecont[start+7+a].split()[3*col+2])
            y = float(filecont[start+7+a].split()[3*col+3])
            z = float(filecont[start+7+a].split()[3*col+4])
            self.deltas.append(np.linalg.norm([x,y,z]))

def get_IR(dataframe, a1, a2, freqmin, freqmax, intmin, intmax, threshold): # a function to get IR values for a pair of atoms at a certain freq and intensity
    IR_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    pair_label = str(a1)+"_"+str(a2)

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        #if True:
        try:
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file)
            if error != "":
                print(error)
                row_i = {'IR_freq_'+str(pair_label): "no data"}
                IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
                continue
            #this changes a1 and a2 (of the form "C1" and "O3") to atomnum_pair (of the form [17, 18])
            atom1 = row[str(a1)]
            atom2 = row[str(a2)]

            #this section finds where all IR frequencies are located in the log file
            frq_len = 0
            frq_end = 0
            for i in range(len(filecont)):
                if frqs_pattern.search(filecont[i]) and frq_len == 1: #subsequent times it finds the pattern, it recognizes the frq_len
                    frq_len = i -3 - frq_start
                if frqs_pattern.search(filecont[i]) and frq_len == 0: #first time it finds the pattern it will set frq_start
                    frq_start = i-3
                    frq_len = 1
                if frqsend_pattern.search(filecont[i]): #finds the end pattern
                    frq_end = i-3

            nfrq = filecont[frq_end-frq_len+1].split()[-1]
            blocks = int((frq_end + 1 - frq_start)/frq_len)
            irdata = []   # list of objects. IR contains: IR.freq, IR.int, IR.deltas = []

            for i in range(0, blocks):
                for j in range(len(filecont[i*frq_len+frq_start].split())):
                    irdata.append(IR(filecont,i*frq_len+frq_start,j,frq_len))

            irout = []
            for i in range(len(irdata)):
                if irdata[i].freq < freqmax and irdata[i].freq > freqmin and irdata[i].int > intmin and irdata[i].int < intmax and irdata[i].deltas[int(atom1)] >= threshold and irdata[i].deltas[int(atom2)] >= threshold:
                        irout = [irdata[i].freq, irdata[i].int]

            #this adds the frequency data from the irout into the new property df
            row_i = {'IR_freq_'+str(pair_label): irout[0]}
            IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire IR frequencies for:', row['log_name'], ".log")
            row_i = {'IR_freq_'+str(pair_label): "no data"}
            IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
    print("IR function has completed for", a1, "and", a2)
    return(pd.concat([dataframe, IR_dataframe], axis = 1))

def get_cone_angle(dataframe, a_list): #DOES NOT MATCH VALUES FROM LITERATURE, WORK IN PROGRESS
    cone_angle_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        if True:
        #try:
            atom_list = []
            for label in a_list:
                atom = row[str(label)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e. "C1")
                atom_list.append(str(atom)) #append that to atom_list to make a list of the form [16, 17, 29]

            log_file = row['log_name']
            streams, errors = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            cone_angle_out = []
            for atom in atom_list:
                cone_angle = ConeAngle(elements, coordinates, int(atom)) #calls morfeus
                cone_angle_out.append(cone_angle)
            cone_angle.print_report()

            row_i = {}
            for a in range(0, len(atom_list)):
                entry = {'cone_angle' + str(a_list[a]) + '(°)': cone_angle_out[a].cone_angle} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                row_i.update(entry)
            cone_angle_dataframe = cone_angle_dataframe.append(row_i, ignore_index=True)
        #except:
        #    print('Unable to acquire cone_angle parameters for:', row['log_name'], ".log")
    print("cone_angle function has completed for", a_list)
    return(pd.concat([dataframe, cone_angle_dataframe], axis = 1))
