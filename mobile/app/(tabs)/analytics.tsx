import React from 'react';
import { StyleSheet, View, Text, ScrollView, TouchableOpacity } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { Svg, Circle, G } from 'react-native-svg';
import Animated, { FadeInDown, ZoomIn } from 'react-native-reanimated';

export default function AnalyticsScreen() {
    return (
        <View style={styles.container}>
            <Animated.View entering={FadeInDown.delay(100).duration(500)} style={styles.header}>
                <TouchableOpacity>
                    <Ionicons name="arrow-back" size={24} color={Colors.text} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Analytics</Text>
                <View style={{ width: 24 }} />
            </Animated.View>

            <ScrollView contentContainerStyle={styles.content}>
                {/* Donut Chart */}
                <Animated.View entering={ZoomIn.delay(300).duration(600)} style={styles.chartContainer}>
                    <Svg height="280" width="280" viewBox="0 0 280 280">
                        <G rotation="-90" origin="140, 140">
                            {/* Background Circle */}
                            <Circle
                                cx="140"
                                cy="140"
                                r="90"
                                stroke="#1C1C1E"
                                strokeWidth="50"
                                fill="transparent"
                            />
                            {/* Progress Circle (88%) - Green */}
                            <Circle
                                cx="140"
                                cy="140"
                                r="90"
                                stroke={Colors.primary}
                                strokeWidth="50"
                                fill="transparent"
                                strokeDasharray="565"
                                strokeDashoffset="68" // 565 * (1 - 0.88)
                                strokeLinecap="butt"
                            />
                            {/* Segment 2 (10%) - Blue - Outer Ring */}
                            <Circle
                                cx="140"
                                cy="140"
                                r="125" // Outer ring
                                stroke="#D0E8F2"
                                strokeWidth="12"
                                fill="transparent"
                                strokeDasharray="785"
                                strokeDashoffset="706" // 10%
                                rotation="320"
                                origin="140, 140"
                                strokeLinecap="round"
                            />
                        </G>
                        {/* Center Text */}
                        <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, justifyContent: 'center', alignItems: 'center' }}>
                            <Text style={styles.percentageText}>%88</Text>
                            <Text style={styles.balanceText}>Balanc Your Expenses</Text>
                        </View>
                    </Svg>

                    {/* Floating Labels */}
                    <View style={{ position: 'absolute', top: 110, left: 0, backgroundColor: '#D0E8F2', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 12 }}>
                        <Text style={{ fontSize: 12, fontWeight: 'bold', color: '#000' }}>%10</Text>
                    </View>
                    <View style={{ position: 'absolute', top: 150, right: 0, backgroundColor: '#F8C8DC', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 12 }}>
                        <Text style={{ fontSize: 12, fontWeight: 'bold', color: '#000' }}>%50</Text>
                    </View>
                    <View style={{ position: 'absolute', bottom: 40, left: 60, backgroundColor: '#D4B483', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 12 }}>
                        <Text style={{ fontSize: 12, fontWeight: 'bold', color: '#000' }}>%35</Text>
                    </View>
                </Animated.View>

                <Animated.View entering={FadeInDown.delay(500).duration(500)} style={styles.statsRow}>
                    <View style={styles.statCard}>
                        <Ionicons name="cart-outline" size={24} color={Colors.text} />
                        <Text style={styles.statLabel}>Products</Text>
                        <Text style={styles.statValue}>%50</Text>
                    </View>
                    <View style={styles.statCard}>
                        <Ionicons name="restaurant-outline" size={24} color={Colors.text} />
                        <Text style={styles.statLabel}>Restorans</Text>
                        <Text style={styles.statValue}>%35</Text>
                    </View>
                    <View style={styles.statCard}>
                        <Ionicons name="medkit-outline" size={24} color={Colors.text} />
                        <Text style={styles.statLabel}>Medicine</Text>
                        <Text style={styles.statValue}>%15</Text>
                    </View>
                </Animated.View>

                <Animated.View entering={FadeInDown.delay(600).duration(500)}>
                    <Text style={styles.sectionTitle}>Transaction</Text>
                    {/* Transaction List Items */}
                    <View style={styles.transactionItem}>
                        <View style={styles.transactionIcon}>
                            <Ionicons name="musical-notes" size={20} color="#fff" />
                        </View>
                        <View style={styles.transactionDetails}>
                            <Text style={styles.transactionName}>Apple music</Text>
                            <Text style={styles.transactionDate}>19 October 2024, 09:15</Text>
                        </View>
                        <Text style={styles.transactionAmount}>- $5.00</Text>
                    </View>
                </Animated.View>
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
        paddingTop: 60,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 24,
        marginBottom: 24,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: Colors.text,
    },
    content: {
        paddingHorizontal: 24,
    },
    chartContainer: {
        alignItems: 'center',
        marginBottom: 40,
        marginTop: 20,
    },
    percentageText: {
        fontSize: 48,
        fontWeight: 'bold',
        color: Colors.text,
    },
    balanceText: {
        color: Colors.textDim,
        textAlign: 'center',
        width: 120,
        fontSize: 14,
        marginTop: 8,
    },
    statsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 40,
    },
    statCard: {
        backgroundColor: Colors.card,
        padding: 16,
        borderRadius: 24,
        alignItems: 'center',
        width: '30%',
        height: 100,
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.12,
        shadowRadius: 5,
        elevation: 3,
    },
    statLabel: {
        color: Colors.textDim,
        marginTop: 12,
        fontSize: 12,
    },
    statValue: {
        color: Colors.text,
        fontWeight: 'bold',
        marginTop: 4,
        fontSize: 16,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: Colors.text,
        marginBottom: 20,
    },
    transactionItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 24,
    },
    transactionIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: Colors.cardElevated,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 16,
        borderWidth: 1,
        borderColor: Colors.border,
    },
    transactionDetails: {
        flex: 1,
    },
    transactionName: {
        color: Colors.text,
        fontWeight: '500',
        fontSize: 16,
    },
    transactionDate: {
        color: Colors.textDim,
        fontSize: 12,
        marginTop: 4,
    },
    transactionAmount: {
        color: Colors.danger,
        fontWeight: '600',
        fontSize: 16,
    },
});
