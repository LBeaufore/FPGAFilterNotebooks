def convert_beams_to_verilog(infilename = "L1Beams.csv", outfilename = "L1Beams_header.vh", lead_antennas = [0,4]):
    with open(infilename, "r") as infile:
        with open(outfilename, "w") as outfile:
            antenna_maxes = {}
            for antenna_idx in range(8):
                antenna_maxes[antenna_idx] = 0
            headerline = next(infile) # Might use in future version
            for beam_idx, line in enumerate(infile):
                line = line.strip().split(",")
                try:
                    elevation = float(line[0])
                    azimuth = float(line[1])
                except Exception as ValueError:
                    break
                delays = []
                for antenna_idx in range(8):
                    delay = int(line[antenna_idx+2])
                    if antenna_maxes[antenna_idx] < delay:
                        antenna_maxes[antenna_idx] = delay
                    outfile.write("`define BEAM_{:d}_ANTENNA_DELAY_{:d} {:d}\n".format(beam_idx, antenna_idx, delay))
                    # outfile.write("`define ")
            for antenna_idx in lead_antennas:
                outfile.write("`define MAX_ANTENNA_DELAY_{:d} {:d}\n".format(antenna_idx, antenna_maxes[antenna_idx]))
            print("Wrote out {:d} beams to \"{:s}\"".format(beam_idx-1, outfilename))