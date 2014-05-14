
import fileinput
import os


for l in fileinput.input():
    try:
        try:
            filenameno = l[:l.rindex(':')]
        except:
            filenameno = ""
        
        if 'mp->' in l:
            name = l[l.rindex('mp->')+4:l.rindex(')')]
        else:
            pass

        vartype = l[l.rindex('sizeof (')+8:]
        vartype = vartype[:vartype.index(')')]
        
        filename = filenameno.split(":")[0]
        
        if "mocl" in name:
            pass
            # name = name[name.index("(void *) ")+9:]
            
            # if not name.startswith('&'):
            #     pass
            
            #     search = "{} \*{}".format(vartype, name)
                
            #     grep = os.popen("grep '{}' {}".format(search, filename)).read().strip()
            #     if not search.replace('\*', '*') in grep:
            #         print "{}:\t{} *{}".format(filenameno, vartype, name)
            #         print "realw? ", os.popen("grep '{}' {}".format(search.replace("int", "realw"), filename)).read().strip()
            #         print "int?", os.popen("grep '{}' {}".format(search.replace("realw", "int"), filename)).read().strip()
            #         raw_input()
            # else:
            #     pass
            #     search = "{} {}".format(vartype, name[1:])
                
            #     grep = os.popen("grep '{}' {}".format(search, filename)).read().strip()
            #     if not search in grep:
            #         print "{}\t{}".format(filenameno, search)
            #         print "realw? ", os.popen("grep '{}' {}".format(search.replace("int", "realw"), filename)).read().strip()
            #         print "int?", os.popen("grep '{}' {}".format(search.replace("realw", "int"), filename)).read().strip()
            #         print
        else:
            pass
            grep = os.popen("grep ' {};' mesh_constants_ocl.h".format(name)).read()
            grep = grep.split(";")[0]
            grep = grep.strip()
            print
            print "FILE:", filenameno
            print "NAME:",name
            print "GREP:",grep
            
            ttype, nname = grep.split(" ")

            assert nname == name
            if ttype != vartype:
                print filenameno, ":\t", vartype, "INSTEAD OF", ttype, nname
                raw_input()
                
            name = "mp->{}".format(name)

                
    except TypeError as e:
        #print "FAIL: {} ({})".format(l, e)
        raise e
