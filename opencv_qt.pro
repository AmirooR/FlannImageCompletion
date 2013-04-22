#-------------------------------------------------
#
# Project created by QtCreator 2011-08-25T12:14:38
#
#-------------------------------------------------

QT       += core

QT       += gui

TARGET = opencv_qt
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    maxflow.cpp \
    LinkedBlockList.cpp \
    graph.cpp \
    GCoptimization.cpp
INCLUDEPATH += /opt/local/include

HEADERS += \
    LinkedBlockList.h \
    graph.h \
    GCoptimization.h \
    energy.h \
    block.h

unix {
        CONFIG += link_pkgconfig
        PKGCONFIG += opencv
}
