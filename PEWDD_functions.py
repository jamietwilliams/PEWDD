import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.lines as mlines
plt.rc('font', family='Times New Roman', size='22')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small') #graph properties

def element_colors():
    
    """
    Defines colours for each metal for use in mass fraction bar and pie charts.
    
    Outputs:
        ele_colors: dictionary of colors
    """
    
    ele_colors = {'H': (0.0, 0.0, 0.502, 1.0), 'He': (0.0, 0.17171856978085354, 0.280638985005767, 1.0),
     'Li': (0.0, 0.3720569011918493, 0.02238446751249512, 1.0), 'Be': (0.0, 0.2128129504625165, 0.41315044684540486, 1.0),
     'B': (0.0, 0.02670630379656419, 0.8696397092170203, 1.0), 'C': (0.0, 0.2766442983467896, 1.0, 1.0),
     'N': (0.0, 0.6642952018454439, 1.0, 1.0), 'O': (0.0, 0.8270829065743943, 1.0, 1.0),
     'F': (0.0, 0.9480825221068819, 0.999858593437375, 1.0),
     'Ne': (0.0, 0.9928728083344707, 0.8198866046418568, 1.0),
     'Na': (0.0, 0.9821726582048133, 0.6656248999599841, 1.0),
     'Mg': (0.0, 0.9853012265331665, 0.4094620992709478, 1.0), 'Al': (0.0, 0.995113491864831, 0.16398122440826918, 1.0),
     'Si': (0.14987017543859635, 0.940799477124183, 0.0, 1.0), 'P': (0.29971537667698656, 0.8522642983467896, 0.0, 1.0),
     'S': (0.4226958093041137, 0.8592963314358, 0.0, 1.0), 'Cl': (0.4770595155709343, 0.9487333965844402, 0.0, 1.0),
     'Ar': (0.5539750508946749, 1.0, 0.07804323729491801, 1.0), 'K': (0.6789045126611666, 1.0, 0.18738030545551568, 1.0),
     'Ca': (0.7859869084610165, 1.0, 0.1873022075496864, 1.0), 'Sc': (0.9109163702275082, 1.0, 0.07796513938908894, 1.0), 
     'Ti': (1.0, 0.9613902782242221, 0.0, 1.0), 'V': (1.0, 0.8936878602807241, 0.0, 1.0),
     'Cr': (1.0, 0.8356572163291544, 0.00843446366782005, 1.0), 'Mn': (1.0, 0.7679547983856564, 0.03798463667820075, 1.0), 
     'Fe': (1.0, 0.6330785099467843, 0.04394123649459788, 1.0), 'Co': (1.0, 0.44038591378263514, 0.021972448979591816, 1.0),
     'Ni': (1.0, 0.24242629051620673, 0.0, 1.0), 'Cu': (1.0, 0.1305415366146459, 0.0, 1.0), 
     'Zn': (1.0, 9.323729491583777e-06, 0.13831931854709198, 1.0), 'Ga': (1.0, 0.0, 0.5556770813243317, 1.0), 
     'Ge': (0.9732819172113288, 0.013357348125890886, 0.9868275515334339, 1.0),
     'As': (0.7841751633986924, 0.10789874032127592, 1.0, 1.0), 
     'Se': (0.6220836601307185, 0.1987140255009108, 0.9952479695703418, 1.0), 
     'Br': (0.7842663623406141, 0.3610698239222828, 0.962296792742598, 1.0), 
     'Other': (0.9235279123414071, 0.5001816608996538, 0.9341238369857747, 1.0), 
     'Rb': (0.9430665513264129, 0.6273442906574394, 0.9508097270280661, 1.0), 
     'Sr': (0.9598139561707035, 0.7363408304498269, 0.9651119184928874, 1.0),
     'Ba': (0.9793525951557094, 0.8635034602076124, 0.9817978085351787, 1.0)}
    
    return ele_colors

def element_dictionary():
    
    """
    Dictionary of different elements and their masses.
    
    Outputs:
        ele_dict: dictionary of elements
    """
    
    ele_dict = {"H":1.008, "He":4.002602, "Li":6.94, "Be":9.0121831, "B":10.81, "C":12.011, "N":14.007, "O":15.999,
                "F":18.99840316, "Ne":20.1797, "Na":22.98976928, "Mg":24.305, "Al":26.9815385, "Si":28.085, "P":30.973762, "S":32.06,
                "Cl":35.45, "Ar":39.948, "K":39.0983, "Ca":40.078, "Sc":44.955908, "Ti":47.867, "V":50.9415, "Cr":51.9961,
                "Mn":54.938044, "Fe":55.845, "Co":58.933194, "Ni":58.6934, "Cu":63.546, "Zn":65.38, "Ga":69.723, "Ge":72.63,
                "As": 74.922, "Se": 78.971,"Br": 79.904,"Kr": 83.798,"Rb": 85.468,"Sr": 87.62, "Ba":137.328}
    
    return ele_dict

def element_database(dataframe, element):
    
    """
    Takes dataframe and extracts data for one element
    Inputs:
        dataframe: database
        element: element you want data from
    Ouputs:
        number_abundance
        number_abundance_error
        accretion_rate_steady
        accretion_rate_in
        accretion_rate_de
        sinking_timescale
        
    Example:
        If you want steady-state accretion rate for Si, pass element="Si" and select third returned list
    """
    
    number_abundance = dataframe["["+element+"/Hx]"].values.tolist()
    number_abundance_error = dataframe["["+element+"/Hx]e"].values.tolist()
    accretion_rate_steady = dataframe["Acc_rate_"+element+"_steady_state"].values.tolist()
    if "Acc_rate_"+element+"_increasing" in dataframe.columns:
        accretion_rate_in = dataframe["Acc_rate_"+element+"_increasing"].values.tolist()
    else: accretion_rate_in = []
    if "Acc_rate_"+element+"_decreasing" in dataframe.columns:
        accretion_rate_de = dataframe["Acc_rate_"+element+"_decreasing"].values.tolist()
    else: accretion_rate_de = []
    if "Sinking_time_"+element in dataframe.columns:
        sinking_timescale = dataframe["Sinking_time_"+element].values.tolist() #takes info from dataframe and converts it to list
    
    return number_abundance, number_abundance_error, accretion_rate_steady, accretion_rate_in, accretion_rate_de, sinking_timescale


def include_dataframe(dataframe, column, values):
    
    """
    Takes dataframe and only keeps rows depending on certain paramters
    Inputs:
        dataframe: database
        column: column in which parameters you want to keep
        values: list of values you want kept
    Output:
        new_dataframe: database with only certain values kept
        
    Example:
        If you want to only have Swan 2019 and Gansicke 2012, pass these in a list under values and the column of "Paper"
    """
    
    df_list = []
    
    for value in values:
        temp_dataframe = dataframe[dataframe[column] == value]
        df_list.append(temp_dataframe)
        
    new_dataframe = pd.concat(df_list)
    return new_dataframe

def mass_fraction_bar_chart(dataframe, wds, trace_limit=0.05, SS_objects=[]):
    
    """
    Creates a bar chart of the mass fraction for a wd
    
    Inputs:
        dataframe: database
        wds: white dwarfs that the bar chart will plot, must include paper as it uses identifier column
        trace_limit: if an element is below this limit, it will be added to the trace group. Used
        for metals that contribute a negligible mass fraction
        SS_objects: which Solar System objects to plot. Choose from: EARTH_A (Bulk Earth, Allegre), EARTH_M (Bulk Earth, Mcdonough), EARTH_C (Core Earth),
        EARTH_S (Silicate Earth), MARS, VENUS, CHONDRITES, SOLAR_L (Lodders), SOLAR_A (Asplund)
    Output:
        plots a bar chart, returns axis
        
    Example:
        To make same plot as left panel of Fig. 9 in Williams et al. (2024), pass:
        mass_fraction_bar_chart(df, ["GD 424 (Keck) Izquierdo 2021", "PG 0843+516 Gansicke 2012",
        "SDSS J1043+0855 Melis & Dufour 2017", "G238-44 Johnson 2022", "WD J2047-1259 Hoskin 2020", 
        "WD 1425+540 (Model 2) Xu 2017"])
    """
    
    check_df = dataframe[dataframe["Identifier"].isin(wds)]
    wd_check = wds
    
    for wd in wd_check:
        if wd not in check_df["Identifier"].values:
            print(wd+" not an identifier in PEWDD")
            wds.remove(wd) #checks that identifier is in PEWDD
    
    dataframe = include_dataframe(dataframe, "Identifier", wds) #only selects wd from database

    wds_included = dataframe["Star"].values.tolist()
    
    for i, wd in enumerate(wds_included):
        if "(" in wd:
            wds_included[i] = wd.split('(')[0] #only keeps name
    
    dataframe_columns = dataframe.columns.tolist()
    acc_total = [col for col in dataframe_columns if col.endswith('_steady_state')]
    total_element_list = [acc.split('_')[2] for acc in acc_total] #identifies necessary elements
    
    remove_elements = ["Total"]
    total_element_list = [ele for ele in total_element_list if ele not in remove_elements]
    elements_present = [] #removes some elements
    
    metal_dict = {"Other": []}
    
    for element in total_element_list:
        metal_dict[element] = [] #creates dictionary with keys being elements - allows multiple bars to be plotted with layers
    
    for i, wd in enumerate(wds):

        total_metal_mass = 0
        trace_element_mass = 0
        
        metal_dict_without_trace = list(metal_dict.keys())[1:]
        
        for element in metal_dict_without_trace:
            mass_fraction_val = dataframe['mass_fraction_'+element].values.tolist()[i]
            if np.isfinite(mass_fraction_val) == True and mass_fraction_val > trace_limit:
                total_metal_mass += mass_fraction_val
                metal_dict[element].append(mass_fraction_val)
                elements_present.append(element) #adds mass fractions to dictionary if above limit
            elif np.isfinite(mass_fraction_val) == True and mass_fraction_val < trace_limit:
                total_metal_mass += mass_fraction_val
                trace_element_mass += mass_fraction_val
                metal_dict[element].append(0) #adds elements to trace if below limit
            elif np.isfinite(mass_fraction_val) == False:
                metal_dict[element].append(0) #ignores if NaN
        
        metal_dict["Other"].append(trace_element_mass) #adds trace to dictionary
    
    elements_present = list(set(elements_present)) #gets only elements present
    
    elements_not_present = list(set(total_element_list).symmetric_difference(set(elements_present))) #gets all elements not present
    
    for ele in elements_not_present:
        metal_dict.pop(ele, None) #removes all non-present elements
        
    ax = plt.figure(figsize=(10,8))
    ax = plt.subplot(1,1,1)
    
    if len(SS_objects) > 0:
        names_m, metal_m = SS_object_bar(ax, SS_objects, trace_limit)
        if len(names_m) > 0:
            df_wd = pd.DataFrame(metal_dict, index=range(len(wds_included)))
            df_m = pd.DataFrame(metal_m, index=range(len(wds_included), len(wds_included)+len(names_m)))
            df_tot = pd.concat([df_m, df_wd])
            df_tot = df_tot.fillna(0)
            metal_dict = df_tot.to_dict('list')
            wds_included = names_m + wds_included #loads in SS_object mass fractions if wanted then adds to wd mass fractions
        
    bottom = np.zeros(len(wds) + len(names_m)) #defines properties of bars
    width = 0.75
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='black', which='both')
    ele_colors = element_colors()
    for element, value in metal_dict.items():
        ax.bar(wds_included, value, width, label=element, bottom=bottom, color=ele_colors[element])
        bottom += value #plots bar chart with layers
        
    ax.set_ylim(0,1)
    ax.set_yticks(np.array(range(0,11))/10)
    ax.legend(loc='upper right',bbox_to_anchor=(1.25, 1.))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.set_ylabel("Mass fraction")
    
    return ax

def SS_object_bar(ax, names, trace_limit):
    """
    Used to plot bar chart of mass fractions of Solar System objects, called from mass_fraction_bar_chart
    
    Inputs:
        ax: bar chart axis
        names: names of Solar System objects
        trace_limit: limit of mass fraction
    """
    
    dataframe = pd.read_csv("/home/jamie/WD_database/SS_object/mass_fractions.csv") #loads SS_object mass fraction dataframe

    names_check = names

    for met in names_check:
        if met not in dataframe["Class"].values:
            print(met+" not an identifier in the Solar System mass fractions")
            names.remove(met) #checks that the identifiers used are valid

    df = include_dataframe(dataframe, "Class", names)
    
    df_col = df.columns.tolist()
    df_col.remove('Class')
    df_col.remove('Name') #gets elements from dataframe
    
    metal_dict = {"Other": []} #defines trace column
    
    for ele in df_col:
        metal_dict[ele.split('_')[1]] = []
    
    for i in range(len(df)):
        trace_mass = []
        for frac in df_col:
            if df[frac].iloc[i] < trace_limit:
                trace_mass.append(df[frac].iloc[i])
                metal_dict[frac.split('_')[1]].append(np.nan) #if under trace limit, is placed in trace column
            else:
                metal_dict[frac.split('_')[1]].append(df[frac].iloc[i]) #if over trace limit
        metal_dict["Other"].append(np.nansum(trace_mass))
        
    metal_dict = {x:y for x,y in metal_dict.items() if np.isnan(y).all() == False} #gets rid of elements that are below trace limit
    
    names_plot = df["Name"].values.tolist()
    
    return names_plot, metal_dict

def mass_fraction_pie_chart(dataframe, wds, layout, trace_limit=0.05, SS_objects = []):
    
    """
    Creates a pie chart of the mass fraction for a wd
    
    Inputs:
        dataframe: database
        wd: wd that the pie chart will plot
        trace_limit: if an element is below this limit, it will be added to the trace group. Used
        for metals that contribute a negligible mass fraction
        layout: list with [x,y] to plot all pie charts together
        SS_objects: which Solar System objects to plot. Choose from: EARTH_A (Bulk Earth, Allegre), EARTH_M (Bulk Earth, Mcdonough), EARTH_C (Core Earth),
        EARTH_S (Silicate Earth), MARS, VENUS, CHONDRITES, SOLAR_L (Lodders), SOLAR_A (Asplund)

    Output:
        plots a pie chart, returns axis
        
    Example:
        To make same plot as right panel of Fig. 9 in Williams et al. (2024), pass:
        mass_fraction_pie_chart(df, ["GD 424 (Keck) Izquierdo 2021", "PG 0843+516 Gansicke 2012",
        "SDSS J1043+0855 Melis & Dufour 2017", "G238-44 Johnson 2022", "WD J2047-1259 Hoskin 2020", 
        "WD 1425+540 (Model 2) Xu 2017"],layout=[3,2])
    """
    
    check_df = dataframe[dataframe["Identifier"].isin(wds)]
    wd_check = wds
    
    for wd in wd_check:
        if wd not in check_df["Identifier"].values:
            print(wd+" not an identifier in PEWDD")
            wds.remove(wd) #checks that identifier is in PEWDD
    
    dataframe = include_dataframe(dataframe, "Identifier", wds) #only selects wd from database
    
    wds_included = dataframe["Star"].values.tolist()
    
    if layout[0] * layout[1] != len(wds) + len(SS_objects):
        print("Layout incorrect")
    
    if len(SS_objects) > 0:
        for i, SS in enumerate(SS_objects):
            ax = plt.subplot(layout[0], layout[1], i+1)
            ele_colors = element_colors()
            
            pie_metal_final, name, elements_final = SS_object_pie(ax, SS, trace_limit)
            
            if pie_metal_final != None:
                ax.pie(pie_metal_final, labels=elements_final, explode=[0.1]*len(pie_metal_final), textprops={'fontsize': 22}, colors=[ele_colors[key] for key in elements_final], normalize=True)
                ax.set_title(name, fontsize=22)
    
    for i, wd in enumerate(wds):
        
        if "(" in wd:
            wds_included[i] = wd.split('(')[0]
        
        dataframe_columns = dataframe.columns.tolist()
        mass_total = [col for col in dataframe_columns if col.startswith('mass_fraction_')]
        element_list = [mass.split('_')[2] for mass in mass_total]
        if "Total" in element_list:
            element_list.remove("Total")
        
        elements_present = [] #identifies elements in wd
        
        total_metal_mass = 0
        pie_metal = []
        trace_element_mass = 0
    
        for j, element in enumerate(element_list):
            
            mass_fraction_val = dataframe['mass_fraction_'+element].iloc[i]
            
            if np.isfinite(mass_fraction_val) == True and mass_fraction_val > trace_limit:
                total_metal_mass += mass_fraction_val
                elements_present.append(element)
                pie_metal.append(mass_fraction_val)
            elif np.isfinite(mass_fraction_val) == True and mass_fraction_val < trace_limit:
                total_metal_mass += mass_fraction_val
                trace_element_mass += mass_fraction_val
                pie_metal.append(0) #checks to see whether there is a value in mass fraction, and adds in if there is
                element_list[j] = ''
            else:
                pie_metal.append(0)
                element_list[j] = ''
                
        element_list.append("Other") #makes labels
        pie_metal.append(trace_element_mass)
                
        pie_metal_fraction = np.array(pie_metal)/total_metal_mass #makes sure fraction adds to one -- may not be needed anymore? Should check
        
        ax = plt.subplot(layout[0], layout[1], len(SS_objects)+i+1)
        ele_colors = element_colors()
        pie_metal_final = []
        elements_final = []
        for j in range(len(element_list)):
            if element_list[j] != '':
                pie_metal_final.append(pie_metal_fraction[j])
                elements_final.append(element_list[j])
        ax.pie(pie_metal_final, labels=elements_final, explode=[0.1]*len(pie_metal_final), textprops={'fontsize': 22}, colors=[ele_colors[key] for key in elements_final], normalize=True)
        ax.set_title(wds_included[i], fontsize=22)
        
    return ax

def SS_object_pie(ax, name, trace_limit):
    """
    Used to plot pie chart of mass fractions of Solar System objects, called from mass_fraction_pie_chart
    Inputs:
        ax: bar chart axis
        names: names of Solar System objects
        trace_limit: limit of mass fraction
    """
    
    dataframe = pd.read_csv("mass_fractions.csv") #loads SS_object mass fraction dataframe

    if name not in dataframe["Class"].values.tolist():
        print(name+" not an identifier in the Solar System mass fractions")
        return None, None, None
    
    df = dataframe[dataframe["Class"] == name]
    
    df_col = df.columns.tolist()
    df_col.remove('Class')
    df_col.remove('Name') #gets elements from dataframe
    
    element_list = []
    
    for ele in df_col:
        element_list.append(ele.split('_')[1])
    
    elements_present = [] #identifies elements in object
    
    total_metal_mass = 0
    pie_metal = []
    trace_element_mass = 0

    for j, element in enumerate(element_list):
        mass_fraction_val = df['X_'+element].values
        
        if np.isfinite(mass_fraction_val) == True and mass_fraction_val > trace_limit:
            total_metal_mass += mass_fraction_val
            elements_present.append(element)
            pie_metal.append(mass_fraction_val)
        elif np.isfinite(mass_fraction_val) == True and mass_fraction_val < trace_limit:
            total_metal_mass += mass_fraction_val
            trace_element_mass += mass_fraction_val
            pie_metal.append(0) #checks to see whether there is a value in mass fraction, and adds in if there is
            element_list[j] = ''
        else:
            pie_metal.append(0)
            element_list[j] = ''
            
    element_list.append("Other") #makes labels
    pie_metal.append(trace_element_mass)
    
    pie_metal_final = []
    elements_final = []
    for j in range(len(element_list)):
        if element_list[j] != '':
            pie_metal_final.append(pie_metal[j][0])
            elements_final.append(element_list[j]) #removes metals with mass fractions under trace limit
     
    name_plot = df["Name"].values.tolist()[0]
     
    return pie_metal_final, name_plot, elements_final

def number_abundance_func(dataframe, reference_element, x_axis_element, y_axis_element,
                        include_column=None, include_values=None,
                        remove_poorly_defined=False, remove_double_arrows = False,
                        point_color='black', errors=True, symbol=".",
                        tracks=False, time_step=1, step_number=5):
    
    """
    Constructs single number abundance plot of the ratio of elements
    
    Not to be used to plot graphs by itself, instead is called in 
    number_abundance_plot function
    
    See Gansicke 2012 Figure 7 for an example of this plot
    
    Inputs:
        dataframe: database (pandas dataframe)
        reference_element: element that you want on denominator (string)
        x_axis_element: element on x-axis (string)
        y_axis_element: element on y-axis (string)
        include_column: column you want to keep (string)
        include_values: values you want to keep from column (list)
        remove_poorly_defined: remove points with 0 errors (Boolean)
        remove_double_arrows_remove: remove points with two arrows (Boolean)
        point_color: color of points on plot (string)
        symbol: symbol used on plot (string)
        errors: whether you want to plot errors (Boolean)
        tracks: whether you want tracks showing evolution of composition over time (Boolean)
        time_step: time interval between each point in Myrs (integer)
        step_number: number of points in track (integer)
    Outputs:
        number abundance plot
        
    Example:
        Want to plot log(O/Si) against log(Fe/Si), define reference_element = "Si", x_axis_element = "O", y_axis_element = "Fe"
    """
    
    dataframe = include_dataframe(dataframe, include_column, include_values)
        
    num_ref, num_ref_err, acc_steady_ref, acc_inc_ref, acc_dec_ref, sink_ref = element_database(dataframe, reference_element)
    num_x, num_x_err, acc_steady_x, acc_inc_x, acc_dec_x, sink_x = element_database(dataframe, x_axis_element)
    num_y, num_y_err, acc_steady_y, acc_inc_y, acc_dec_y, sink_y = element_database(dataframe, y_axis_element)
    ele_dict = element_dictionary() #gets all element data from dataframe
    
    acc_steady_ref, acc_steady_x, acc_steady_y = np.array(acc_steady_ref).astype(np.float), np.array(acc_steady_x).astype(np.float), np.array(acc_steady_y).astype(np.float)
    num_ref_err, num_x_err, num_y_err = np.array(num_ref_err).astype(np.float), np.array(num_x_err).astype(np.float), np.array(num_y_err).astype(np.float)
    sink_ref, sink_x, sink_y = np.array(sink_ref).astype(np.float), np.array(sink_x).astype(np.float), np.array(sink_y).astype(np.float)
    
    ref_nan, x_nan, y_nan = np.isfinite(acc_steady_ref), np.isfinite(acc_steady_x), np.isfinite(acc_steady_y)
    full_nan = ref_nan & x_nan & y_nan
    acc_steady_ref, acc_steady_x, acc_steady_y = acc_steady_ref[full_nan], acc_steady_x[full_nan], acc_steady_y[full_nan]
    num_ref_err, num_x_err, num_y_err = num_ref_err[full_nan], num_x_err[full_nan], num_y_err[full_nan]
    sink_ref, sink_x, sink_y = sink_ref[full_nan], sink_x[full_nan], sink_y[full_nan]#finds NaN values and removes from all subsequent calculations
    
    number_abundance_x_unlogged = (acc_steady_x/acc_steady_ref) * (ele_dict[reference_element]/ele_dict[x_axis_element])
    number_abundance_y_unlogged = (acc_steady_y/acc_steady_ref) * (ele_dict[reference_element]/ele_dict[y_axis_element])
    number_abundance_x = np.log10(number_abundance_x_unlogged) 
    number_abundance_y = np.log10(number_abundance_y_unlogged)
    
    #number abundance defined as:
    #log[ N(X)/N(ref) ] = log[ M_dot(X)/M_dot(ref) * A(ref)/A(X) ]
    #as seen in equation 2 of Gansicke et al. 2012
    
    upper_limit_down = np.full(len(acc_steady_ref), False, dtype=bool)
    upper_limit_left = np.full(len(acc_steady_ref), False, dtype=bool)
    upper_limit_up_right = np.full(len(acc_steady_ref), False, dtype=bool) #identifies when errors are upper limits
    arrow_length = 0.25
    
    pop_list = []
    
    if remove_poorly_defined == True:
        for i in range(len(num_ref_err)):
            if num_ref_err[i] > -0.001 and num_ref_err[i] < 0.001:
                pop_list.append(i)
            elif num_x_err[i] > -0.001 and num_x_err[i] < 0.001:
                pop_list.append(i)
            elif num_y_err[i] > -0.001 and num_y_err[i] < 0.001:
                pop_list.append(i) #removes any poorly defined quantities - e.g. has error of 0
    
    for i, val in enumerate(num_ref_err):
        upper_limit_up_right[i] = (num_ref_err[i] == -1.0)
        upper_limit_left[i] = (num_x_err[i] == -1.0)
        upper_limit_down[i] = (num_y_err[i] == -1.0) #creates list of True when there is an upper limit and False elsewhere
            
    num_ref_err[num_ref_err < 0] += 1
    num_x_err[num_x_err < 0] += 1
    num_y_err[num_y_err < 0] += 1 #removes upper limit of -1 from errors
    
    number_abundance_x_err = np.sqrt((num_ref_err)**2 + (num_x_err)**2)
    number_abundance_y_err = np.sqrt((num_ref_err)**2 + (num_y_err)**2)
    
    for i, val in enumerate(num_ref_err):
        if upper_limit_up_right[i] == True or upper_limit_left[i] == True or upper_limit_down[i] == True:
            number_abundance_x_err[i] = arrow_length
            number_abundance_y_err[i] = arrow_length #makes length of arrows all the same
        if upper_limit_up_right[i] == True and upper_limit_down[i] == True and upper_limit_left[i] == True:
            pop_list.append(i) #if upper limits in both directions (e.g. left AND right, removes)
        if remove_double_arrows == True and upper_limit_up_right[i] == True:
            pop_list.append(i)
        if remove_double_arrows == True and upper_limit_down[i] == True and upper_limit_left[i] == True:
            pop_list.append(i)
            
    number_abundance_x, number_abundance_y = np.delete(number_abundance_x, pop_list), np.delete(number_abundance_y, pop_list) #removes completely unbound quantities
    number_abundance_x_err, number_abundance_y_err = np.delete(number_abundance_x_err, pop_list), np.delete(number_abundance_y_err, pop_list)
    upper_limit_left, upper_limit_up_right, upper_limit_down = np.delete(upper_limit_left, pop_list), np.delete(upper_limit_up_right, pop_list), np.delete(upper_limit_down, pop_list)
    
    if tracks == True:
        time_step_s = time_step * 10**6 * 365.25 * 24 * 60 * 60
        
        track_ref = np.exp((time_step_s/sink_ref))
        track_x = np.exp((time_step_s/sink_x))
        track_y = np.exp((time_step_s/sink_y))
        
        total_step_x = []
        total_step_y = []
        
        for i in range(step_number):
            total_step_x.append(np.log10(number_abundance_x_unlogged*((track_x)/track_ref)**i))
            total_step_y.append(np.log10(number_abundance_y_unlogged*((track_y)/track_ref)**i))
        plt.plot(total_step_x, total_step_y, color=point_color)
        plt.scatter(total_step_x, total_step_y, color=point_color, marker=symbol, s=50)
    
    plt.scatter(number_abundance_x, number_abundance_y, color=point_color, marker=symbol, s=250)
    
    if errors == True:
        plt.errorbar(number_abundance_x, number_abundance_y,
                    xerr = number_abundance_x_err, yerr = number_abundance_y_err,
                    color = point_color, fmt=symbol, 
                    xuplims = upper_limit_left, xlolims = upper_limit_up_right,
                    lolims = upper_limit_up_right, uplims = upper_limit_down) #plots graph if you want errors

def number_abundance_plot(dataframe, reference_element, x_axis_element, y_axis_element,
                         include_column, include_values=None,
                         remove_poorly_defined=False, remove_double_arrows = False,
                         point_colors="standard", point_symbols="standard", errors=True,
                         tracks=False, time_steps=1, step_number=5, legend=True,
                         plot_meteorite=False, meteorite_fileloc='/home/jamie/WD_database/meteorite',
                         meteorite_classes="all"):
    
    """
    Constructs number abundance plot of the ratio of elements
    
    Inputs:
        dataframe: database (pandas dataframe)
        reference_element: element that you want on denominator (string)
        x_axis_element: element on x-axis (string)
        y_axis_element: element on y-axis (string)
        include_column: column to choose WDs from (string)
        include_values: WDs to be used (list)
        remove_poorly_defined: remove points with 0 errors (Boolean)
        remove_double_arrows_remove: remove points with two arrows (Boolean)
        point_colors: color of points on plot (list)
        point_symbols: symbols used on plot (list)
        error_list: whether you want to plot errors for (Boolean)
        track_list: whether you want to plot tracks for (Boolean list)
        time_step_list: the time steps of tracking (integer)
        step_number_list: number of steps of tracking (integer)
        legend: include a legend or not (Boolean)
        plot_meteorite: whether to plot Solar System objects
        meteorite_fileloc: file location of meteorite .csv files
        meteorite_classes: which meteorite classes to plot
    Outputs:
        number abundance plot
        
    Example:
        Plot of Fig. 3 from Williams et al. 2024, plotting O/Si against Fe/Si:
        number_abundance_plot(df, "Si", "O", "Fe", include_column="Identifier", 
        include_values=["GD 424 (Keck) Izquierdo 2021", "WD 1415+234 Doyle 2023",
        "GD 362 Xu 2013", "Ton 345 Wilson 2015", "WD J2047-1259 Hoskin 2020",
        "SDSS J0956+5912 (Overshoot) Hollands 2022"], track_list = True,
        time_step_list=1e-2, step_number_list=5, legend=True, 
        plot_meteorite=True, point_colors=['C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
        
        Plot all H and He atmosphere white dwarfs with abundances C/Mg and O/Mg
        number_abundance_plot(df, "Mg", "C", "O", include_column="atmosphere",
        include_values=["H", "He"], remove_double_arrows=True, point_colors=["blue", "red"],
        plot_meteorite=True)
    """
    
    if point_colors == "standard":
        colors = []
        for i in range(len(include_column)):
            colors.append('#%06X' % randint(0, 0xFFFFFF)) #generates random colour if one not specified
    else:
        colors = point_colors
        
    if point_symbols == "standard":
        symbols = ["."]*len(include_values)
    else:
        symbols = point_symbols
        
    step_number = step_number + 1
        
    ax = plt.figure(figsize=(10,8))
    ax = plt.subplot(1,1,1)
    
    for j, val in enumerate(include_values):
        number_abundance_func(dataframe, reference_element, x_axis_element, y_axis_element,
                                  include_column=include_column, include_values=[val],
                                  remove_poorly_defined = remove_poorly_defined, point_color=colors[j], errors=errors, symbol=symbols[j],
                                  tracks=tracks, time_step=time_steps, step_number=step_number) #for each inclusion, plots a graph with a different colour
    
    if legend == True:
        legend_list = []
        total_patch = []
        for ls in include_values:
            for lab in ls:
                legend_list.append(lab)
        for i in range(len(include_values)):
            total_patch.append(mlines.Line2D([], [], color=colors[i], linewidth=0, marker=symbols[i], label=include_values[i], markersize=10))
            
        ax.legend(handles=total_patch, fontsize=18) #plots legend with correct symbols and colors
        
    if plot_meteorite == True:
        plot_meteorites(reference_element, x_axis_element, y_axis_element, ax=ax, include_classes=meteorite_classes)
    
    ax.set_xlabel('log('+x_axis_element+'/'+reference_element+')')
    ax.set_ylabel('log('+y_axis_element+'/'+reference_element+')')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, direction='in', which='both')
    
    return ax

def plot_meteorites(ref_element, x_element, y_element, ax,
                    file_loc='/home/jamie/WD_database/meteorite',
                    include_classes="all",cmap="plasma"):
    
    """
    Plots meteorites on number abundance plot
    
    Inputs:
        ref_element: element that you want on denominator (string)
        x_element: element on x-axis (string)
        y_element: element on y-axis (string)
        ax: axis to plot meteorites on
        file_loc: file location of meteorite .csv files
        include_classes: which meteorite classes to include
        cmap: meteorite color map
    """
    
    database = pd.read_csv('/home/jamie/WD_database/meteorite/meteorite_database_'+ref_element+'.csv')
    
    NUM_COLORS = 24
    cm = plt.get_cmap(cmap)
    
    if include_classes == "all":
        include_classes = database["Class"].unique()
        
    df = database[["Names", "Class", "["+x_element+"/"+ref_element+"]",
                   "["+y_element+"/"+ref_element+"]"]]
    
    special_classes = ["CHONDRITES", "SOLAR_L", "SOLAR_A", "EARTH_A", "EARTH_S", "EARTH_C", "EARTH_M", "MARS", "VENUS"]
    
    for j, cl in enumerate(include_classes):
        df_temp = df[df["Class"]==cl]
        x_val = df_temp["["+x_element+"/"+ref_element+"]"]
        y_val = df_temp["["+y_element+"/"+ref_element+"]"]
        #c = df_temp["Color"].values.tolist()
        if cl not in special_classes:
            ax.scatter(x_val, y_val, color=[cm(1.*j/NUM_COLORS)], s = 5, zorder=0)
        
        elif cl == 'CHONDRITES':
            ax.scatter(x_val, y_val, color='black', s = 100, marker="h", zorder=10)
        elif cl == "SOLAR_L": #or cl == "SOLAR_A":
            ax.scatter(x_val, y_val, color='black', s = 100, marker="*", zorder=10)
        #elif cl == "EARTH_C" or cl == "EARTH_A" or cl == "EARTH_M" or cl == "EARTH_S":
        #    ax.scatter(x_val, y_val, color=cols[j], s = 50, marker="X")
        elif cl == "EARTH_A":# or cl == "EARTH_M":
            ax.scatter(x_val, y_val, color='black', s = 100, marker="X", zorder=10)
        elif cl == "EARTH_C":
            ax.scatter(x_val, y_val, color='black', s = 100, marker="D", zorder=10)
        elif cl == "EARTH_S":
            ax.scatter(x_val, y_val, color='black', s = 100, marker="s", zorder=10)
        elif cl == "VENUS":
            ax.scatter(x_val, y_val, color='black', s = 100, marker="d")
        elif cl == "MARS":
            ax.scatter(x_val, y_val, color='black', s = 100, marker="P")