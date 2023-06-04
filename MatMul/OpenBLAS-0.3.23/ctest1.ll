; ModuleID = 'ctest1.c'
target datalayout = "e-p:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define internal void @pgCplus_compiled.() noinline {
L.entry:
	ret void
}


define dso_local signext i32 @hogehoge() #0 mustprogress !dbg !15 {
L.entry:

	ret i32 0, !dbg !18
}

attributes #0 = { null_pointer_is_valid }

declare i32 @__gxx_personality_v0(...)

; Named metadata
!llvm.module.flags = !{ !1, !2 }
!llvm.dbg.cu = !{ !10 }

; Metadata
!1 = !{ i32 2, !"Dwarf Version", i32 4 }
!2 = !{ i32 2, !"Debug Info Version", i32 3 }
!3 = !DIFile(filename: "ctest1.c", directory: "/large/___HOME___/area/ntosato/lecture2/MPIMatMul/cublas_new/OpenBLAS-0.3.23")
; !4 = !DIFile(tag: DW_TAG_file_type, pair: !3)
!4 = !{ i32 41, !3 }
!5 = !{  }
!6 = !{  }
!7 = !{ !15 }
!8 = !{  }
!9 = !{  }
!10 = distinct !DICompileUnit(file: !3, language: DW_LANG_C_plus_plus, producer: " NVC++ 23.3-0", enums: !5, retainedTypes: !6, globals: !8, emissionKind: FullDebug, imports: !9)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{ !11 }
!13 = !DISubroutineType(types: !12)
!14 = !{  }
!15 = distinct !DISubprogram(file: !3, scope: !10, name: "hogehoge", line: 1, type: !13, spFlags: 8, unit: !10, scopeLine: 1)
!16 = !DILocation(line: 1, column: 1, scope: !15)
!17 = !DILexicalBlock(file: !3, scope: !15, line: 1, column: 1)
!18 = !DILocation(line: 1, column: 1, scope: !17)
