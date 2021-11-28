import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.Domain;
import burlap.mdp.core.state.State;

import static burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor.addFloor;

public class BlockDudeLevelV2 {


    public static State getLevel4(Domain domain) {

        int[][] map = new int[25][25];

        BlockDudeLevelConstructor.wallSegment(map, 2, 24, 0);
        BlockDudeLevelConstructor.wallSegment(map, 0, 24, 24);
        BlockDudeLevelConstructor.floorSegment(map, 0, 24, 24);
        BlockDudeLevelConstructor.floorSegment(map, 0, 24, 0);

        addFloor(map);

        map[3][1] = 1;
        map[3][2] = 1;
        map[7][1] = 1;
        map[11][1] = 1;
        map[11][2] = 1;


        BlockDudeState s = new BlockDudeState(
                new BlockDudeAgent(15, 1, 1, false),
                new BlockDudeMap(map),
                BlockDudeCell.exit(0, 1),
                BlockDudeCell.block("b0", 9, 1),
                BlockDudeCell.block("b1", 13, 1)
        );

        return s;
    }
}
