// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project)
{
    project.paths.construct_build_prefix()

    String compiler = platform.jenkinsLabel.contains('hip-clang') ? 'hipcc' : 'hcc'
    String hip = platform.jenkinsLabel.contains('hip-clang') ? '--hip-clang' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/lib CXX=/opt/rocm/bin/${compiler} ${project.paths.build_command} ${hip}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter, gfilter2)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)    
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter2}
                """
    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    if(platform.jenkinsLabel.contains('hip-clang'))
    {
        packageCommand = null
    }
    else
    {
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release",true,true)
        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
    }
}

return this
