/* peakrun.c - run a command, then report its peak working set and wall time.
 *   peakrun OUT.txt cmd args...   (stdout/stderr of the command pass through) */
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: peakrun OUT.txt cmd args...\n"); return 2; }
    char cmd[32768] = { 0 };
    for (char *p = argv[2]; *p; p++) if (*p == '/') *p = '\\';   /* CreateProcess wants a Windows path */
    for (int i = 2; i < argc; i++) {
        strcat(cmd, "\"");
        strcat(cmd, argv[i]);
        strcat(cmd, i + 1 < argc ? "\" " : "\"");
    }
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "cannot start: %s\n", cmd);
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    QueryPerformanceCounter(&t1);
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(pi.hProcess, &pmc, sizeof pmc);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    FILE *o = fopen(argv[1], "w");
    if (o) {
        fprintf(o, "peak_working_set_gb %.3f\nwall_s %.1f\nexit %lu\n", pmc.PeakWorkingSetSize / 1e9,
                (double)(t1.QuadPart - t0.QuadPart) / f.QuadPart, (unsigned long)code);
        fclose(o);
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)code;
}
