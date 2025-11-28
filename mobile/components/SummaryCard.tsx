import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { Colors } from '@/constants/Colors';
import { Ionicons } from '@expo/vector-icons';

interface SummaryCardProps {
    title: string;
    amount: string;
    icon: keyof typeof Ionicons.glyphMap;
    color: string;
    trend?: string;
}

export function SummaryCard({ title, amount, icon, color, trend }: SummaryCardProps) {
    return (
        <View style={styles.container}>
            <View style={[styles.iconContainer, { backgroundColor: color + '20' }]}>
                <Ionicons name={icon} size={20} color={color} />
            </View>
            <View style={styles.content}>
                <Text style={styles.title}>{title}</Text>
                <Text style={styles.amount}>{amount}</Text>
                {trend && <Text style={styles.trend}>{trend}</Text>}
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        backgroundColor: Colors.card,
        borderRadius: 20,
        padding: 16,
        width: '31%',
        justifyContent: 'space-between',
        borderWidth: 1,
        borderColor: Colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 6,
        elevation: 4,
    },
    iconContainer: {
        width: 36,
        height: 36,
        borderRadius: 18,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 12,
    },
    content: {
        gap: 4,
    },
    title: {
        color: Colors.textDim,
        fontSize: 12,
        fontWeight: '500',
    },
    amount: {
        color: Colors.text,
        fontSize: 16,
        fontWeight: 'bold',
    },
    trend: {
        color: Colors.primary,
        fontSize: 10,
        fontWeight: '600',
    },
});
