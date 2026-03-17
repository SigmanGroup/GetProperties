# test_2

## Notes

The SMARTS string for the original property acquisition was

'''python
substructure = Chem.MolFromSmarts('[*][#7](c1[c]([H])[c]([H])[c]([H])[c]([H])[c]1[H])c1ccccc1')
'''

These were the assignments from the mapped atoms. Some of the mapped atoms are duplicated because of a few files that
had multiple matches (prod_3a, prod_5a, prod_5i). All conformers of prod_5l failed because there was no substructure.

'''python
atom_labels = {'log_name': 'log_name',
                0: 'HC',
                1: 'N',
                2: 'Cb',
                13: 'Ca',
                3: 'Cb1',
                11: 'Cb2',
                14: 'Ca1',
                18: 'Ca2'
}
'''

This is an overview of the multiple matches from the original.

'''
prod_3a_1.log
    atom_1 - 14 - H or C atom on the nitrogen   (HC)    checked
    atom_2 - 2 - Nitrogen atom                  (N)     checked
    atom_3 - 1 - Carbon ipso to nitrogen        (Cb)    checked
    atom_4 - 13 - Ortho 1                       (Cb1)   checked
atom_5 - 24 - H attached to ortho 1
atom_6 - 12 - Meta 1
atom_7 - 23 - H attached to meta 1
atom_8 - 11 - Para
atom_9 - 22 - H attached to para
atom_10 - 10 - Meta 2
atom_11 - 21 - H attached to meta 2
    atom_12 - 9 - Ortho 2                       (Cb2)   checked
atom_13 - 20 - H attached to Ortho 2
        atom_14 - 3 - Carbon ipso to nitrogen   (Ca)    checked - this is ipso of the unchanging phenyl group
        atom_15 - 8 - Ortho 1a                  (Ca1)   checked - this is the ortho of the unchanging phenyl group
atom_16 - 7 - Meta 1a
atom_17 - 6 - Para 2
atom_18 - 5 - Meta 2a
    atom_19 - 4 - Ortho 2a                      (Ca2)   checked - this is the other ortho of the unchanging phenyl group
atom_20 - 14 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_21 - 2 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_22 - 3 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_23 - 8 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_24 - 19 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_25 - 7 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_26 - 18 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_27 - 6 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_28 - 17 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_29 - 5 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_30 - 16 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_31 - 4 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_32 - 15 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_33 - 1 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_34 - 13 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_35 - 12 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_36 - 11 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_37 - 10 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
atom_38 - 9 - REPEAT FROM ABOVE BECAUSE SYMMETRIC (ONLY ASSIGNED FOR prod_3a)
'''

## Computed differences between old and new properties

The difference between the old and new properties were computed. For one file, the extracted ZPE, T\*S, and T\*qh_S were
different by around 25%. After inspecting the file, the new extracted ZPE of 0.22768 Hartree/particle is correct. It is
likely that the old version has a bug when extracting this value from the compressed stream.

'''python
prod_3i_1.log   ZPE(Hartree)    new: 0.22768    old: 0.45536    diff: -0.2277 (50.0 %)
prod_3i_1.log   T*S         new: 0.05013        old: 0.06546    diff: -0.0153 (23.42 %)
prod_3i_1.log   T*qh_S      new: 0.04867        old: 0.06254    diff: -0.0139 (22.18 %)
dihedral_Cb_N_Ca_Ca1 (°) vs dihedral_Cb_N_Ca_Ca1(°)     Total absolute difference: 0.00 ax abs: 0.00
dihedral_Cb_N_Ca_Ca2 (°) vs dihedral_Cb_N_Ca_Ca2(°)     Total absolute difference: 0.00 ax abs: 0.00
dihedral_Ca_N_Cb_Cb1 (°) vs dihedral_Ca_N_Cb_Cb1(°)     Total absolute difference: 0.00 ax abs: 0.00
dihedral_Ca_N_Cb_Cb2 (°) vs dihedral_Ca_N_Cb_Cb2(°)     Total absolute difference: 0.00 ax abs: 0.00
plane_angle_Ca-Ca1-Ca2_Cb-Cb1-Cb2 (°) vs planeangle_Ca_Ca1_Ca2_&_Cb_Cb1_Cb2(°)  Total absolute difference: 0.00 ax abs: 0.00
'''
