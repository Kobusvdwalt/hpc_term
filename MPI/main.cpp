#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
    int proccessCount, processId;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proccessCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    printf("P count : %d\n", proccessCount);
    printf("This rank : %d\n", processId);
    
    
    if (processId == 0) {
        double startTime = MPI_Wtime();
        char message[] = "hos";
        char incomingMessage[3];
        MPI_Status status;
        for (int i=1; i < proccessCount; i ++) {
            // Send a hos message to each other process
            MPI_Send(message, 3, MPI_CHAR, i, 0, MPI_COMM_WORLD);

            // Wait for their reply
            MPI_Recv(incomingMessage, 3, MPI_CHAR, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);
            printf("My id : %d\n", processId);
            printf("Message from : %d\n", status.MPI_SOURCE);
            printf("Message : ");
            for (int c=0; c < 3;c ++) {
                printf("%c", incomingMessage[c]);
            }
            printf("\n\n");
        }
        double endTime = MPI_Wtime();
        printf("Total time : %lf\n", endTime - startTime);
        printf("Time resolution : %lf\n", MPI_Wtick());
    }
    if (processId != 0) {
        char message[] = "awe";
        char incomingMessage[3];
        MPI_Status status;
        // Receive message from rank 0
        MPI_Recv(incomingMessage, 3, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        printf("My id : %d\n", processId);
        printf("Message from : %d\n", status.MPI_SOURCE);
        int inCount = 0;
        MPI_Get_count(&status, MPI_CHAR, &inCount);
        printf("Count from : %d\n", inCount);
        
        printf("Message : ");
        for (int c=0; c < 3;c ++) {
            printf("%c", incomingMessage[c]);
        }

        float f = 0;
        for (int i=0; i < 100000000; i ++) {
            f += 1.24;
            f *= 0.9;
        }
        float a = 0.4 + f;
        
        printf("%lf\n\n", a);

        // Send awe message back
        MPI_Send(message, 3, MPI_CHAR, status.MPI_SOURCE, 4, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}